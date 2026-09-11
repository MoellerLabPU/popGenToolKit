"""Unit tests for the permutation orchestrator (``run_permutations``).

These lock in the *generate-mode* leaf contract introduced when per-pair sheet
generation moved from a Python ``subprocess`` loop into the in-DAG
``permute_metadata`` rule:

  * each leaf config carries ``permutation.seed`` (singular) and NOT
    ``permuted_metadata_dir`` (so it is generate mode, not BYO);
  * each leaf runs in its own isolated working dir (``perm_<seed>``);
  * ``metadata_path`` is pinned absolute for the rule;
  * a dry run writes nothing into the output tree (no config/TSV litter).

``execute_workflow`` is monkeypatched so no Snakemake/cluster work happens.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import alleleflux.workflow as wf


def _make_config(tmp_path: Path, seeds):
    """Minimal config that run_permutations needs (generate mode)."""
    md = tmp_path / "metadata.tsv"
    md.write_text("sample_id\tsubjectID\tgroup\ts1\tu1\tA\n")
    return {
        "permutation": {"enabled": True, "seeds": seeds, "unit": "subjectID"},
        "output": {"root_dir": str(tmp_path / "out")},
        "input": {"metadata_path": str(md), "reuse_from": str(tmp_path)},
        "analysis": {"groups_combinations": [{"treatment": "A", "control": "B"}]},
    }, md


def test_leaf_config_carries_seed_not_byo(tmp_path, monkeypatch):
    """Each leaf is a generate-mode leaf: seed set, no permuted_metadata_dir."""
    captured = []

    def fake_execute_workflow(*, config_file, working_dir, **kwargs):
        captured.append(
            {"leaf": yaml.safe_load(open(config_file)), "working_dir": working_dir}
        )
        return 0

    monkeypatch.setattr(wf, "execute_workflow", fake_execute_workflow)

    config, md = _make_config(tmp_path, seeds=[1, 2])
    rc = wf.run_permutations(config, working_dir=str(tmp_path), dry_run=False)

    assert rc == 0
    assert len(captured) == 2  # one leaf per seed
    assert sorted(c["leaf"]["permutation"]["seed"] for c in captured) == [1, 2]

    for c in captured:
        perm = c["leaf"]["permutation"]
        assert perm["enabled"] is True
        # Generate mode — must NOT carry the BYO marker.
        assert "permuted_metadata_dir" not in perm
        seed = perm["seed"]
        seed_dir = tmp_path / "out" / "permuted" / f"perm_{seed}"
        # Output rooted at the per-seed dir, and the leaf runs there (isolated
        # .snakemake/ — the working_dir isolation fix).
        assert c["leaf"]["output"]["root_dir"] == str(seed_dir)
        assert c["working_dir"] == str(seed_dir)
        # metadata pinned absolute for the rule.
        assert c["leaf"]["input"]["metadata_path"] == str(md)
        # within-group tests suppressed by default on a null run.
        assert c["leaf"]["analysis"]["run_within_group_tests"] is False


def test_dry_run_leaves_output_tree_pristine(tmp_path, monkeypatch):
    """A dry run must not write the leaf config/TSVs into output.root_dir."""
    working_dirs = []

    def fake_execute_workflow(*, config_file, working_dir, **kwargs):
        # The config must exist *during* the call (it is in a temp dir)...
        assert Path(config_file).exists()
        working_dirs.append(working_dir)
        return 0

    monkeypatch.setattr(wf, "execute_workflow", fake_execute_workflow)

    config, _ = _make_config(tmp_path, seeds=[7])
    out_root = tmp_path / "out"
    rc = wf.run_permutations(config, working_dir=str(tmp_path), dry_run=True)

    assert rc == 0
    # ...but nothing should land under output.root_dir on a dry run.
    assert not out_root.exists() or not any(out_root.rglob("*"))
    # The leaf ran in a throwaway temp dir, not under output.root_dir.
    assert str(out_root) not in working_dirs[0]


def test_stops_on_first_failure(tmp_path, monkeypatch):
    """A non-zero leaf exit halts orchestration and is propagated."""
    calls = []

    def fake_execute_workflow(*, config_file, working_dir, **kwargs):
        seed = yaml.safe_load(open(config_file))["permutation"]["seed"]
        calls.append(seed)
        return 1 if seed == 1 else 0  # first seed fails

    monkeypatch.setattr(wf, "execute_workflow", fake_execute_workflow)

    config, _ = _make_config(tmp_path, seeds=[1, 2, 3])
    rc = wf.run_permutations(config, working_dir=str(tmp_path), dry_run=False)

    assert rc == 1
    assert calls == [1]  # stopped after the first failure; 2 and 3 not attempted


def test_unlock_cascades_to_started_leaves(tmp_path, monkeypatch):
    """--unlock on an orchestrator clears each started leaf's working dir too."""
    unlocked = []

    def fake_run_snakemake(cmd, *args, **kwargs):
        unlocked.append(cmd)
        return 0

    monkeypatch.setattr(wf, "run_snakemake", fake_run_snakemake)

    config, _ = _make_config(tmp_path, seeds=[1, 2, 3])
    out_root = tmp_path / "out"
    # Seeds 1 and 3 "started" — they have a persisted leaf config + .snakemake/.
    # Seed 2 never ran (no leaf dir) and must be skipped.
    for seed in (1, 3):
        seed_dir = out_root / "permuted" / f"perm_{seed}"
        (seed_dir / ".snakemake").mkdir(parents=True)
        (seed_dir / f"config_perm_{seed}.yml").write_text("placeholder: true\n")

    rc = wf._unlock_permutation_leaves(
        config, working_dir=str(tmp_path), snakefile="Snakefile"
    )

    assert rc == 0
    # One unlock command per started leaf, pointed at that leaf's dir.
    assert sum("perm_1" in c for c in unlocked) == 1
    assert sum("perm_3" in c for c in unlocked) == 1
    assert not any("perm_2" in c for c in unlocked)
    assert all("--unlock" in c for c in unlocked)


def test_unlock_noop_for_non_orchestrator(tmp_path, monkeypatch):
    """A leaf/BYO config (seed set) is not an orchestrator — no cascade."""
    called = []
    monkeypatch.setattr(wf, "run_snakemake", lambda *a, **k: called.append(a) or 0)

    config, _ = _make_config(tmp_path, seeds=[1])
    config["permutation"]["seed"] = 1  # marks this as a leaf, not an orchestrator

    rc = wf._unlock_permutation_leaves(
        config, working_dir=str(tmp_path), snakefile="Snakefile"
    )

    assert rc == 0
    assert called == []  # nothing unlocked
