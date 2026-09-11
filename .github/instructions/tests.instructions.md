---
applyTo: "tests/**/*.py"
---

# Test Conventions for AlleleFlux

## Framework & Commands

- **Primary framework**: `unittest` (most test files) with some `pytest`-style tests
- **Run all tests**: `pytest tests/ -v --tb=short`
- **Run by module**: `pytest tests/statistics/`
- **With coverage**: `pytest tests/ --cov=alleleflux --cov-report=html`
- CI runs tests via GitHub Actions (see [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml))

## File Structure

Mirror the `alleleflux/scripts/` directory structure:

```
tests/
├── statistics/      → tests for alleleflux/scripts/statistics/
├── preprocessing/   → tests for alleleflux/scripts/preprocessing/
├── analysis/        → tests for alleleflux/scripts/analysis/
├── evolution/       → tests for alleleflux/scripts/evolution/
├── accessory/       → tests for alleleflux/scripts/accessory/
├── utilities/       → tests for alleleflux/scripts/utilities/
└── visualization/   → tests for alleleflux/scripts/visualization/
```

Each subdirectory has an `__init__.py`. Test file naming: `test_<module_name>.py`.

## Naming Conventions

- **Classes**: `Test<ComponentName>` (e.g., `TestRunPairedTests`, `TestQualityControl`)
- **Methods**: `test_<behavior_description>` (e.g., `test_identical_values_edge_case`, `test_zero_coverage_warning`)
- **Constants**: `NUCLEOTIDES = ["A", "T", "G", "C"]`
- Give each edge case its own dedicated test method with a descriptive name

## Fixtures & Data Setup

- Use `setUp()` / `tearDown()` for test class fixtures (no shared `conftest.py`)
- Create test data inline as dicts or `pd.DataFrame()` — no external factory libraries
- Temporary files: use `tempfile.TemporaryDirectory()` or `tempfile.NamedTemporaryFile()` with cleanup in `tearDown()` or `try/finally`
- For I/O tests, create helper methods like `create_temp_profile()` on the test class

```python
class TestExample(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = pd.DataFrame({
            "contig": ["c1", "c1"], "position": [1, 2],
            "A": [10, 0], "T": [0, 10], "G": [0, 0], "C": [0, 0],
            "total_coverage": [10, 10]
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
```

## Mocking Patterns

- Mock `multiprocessing.Pool` for parallel processing tests
- Mock statistical backends: `scipy.stats`, `statsmodels.formula.api.mixedlm`, R calls via rpy2
- Mock I/O: `pandas.read_csv`, `pandas.DataFrame.to_csv`, `os.makedirs`
- Mock `pysam` objects (AlignmentFile, FastaFile) for BAM/FASTA tests
- Patch logger to verify warning/error messages:
  ```python
  @patch("alleleflux.scripts.module.logger")
  def test_warning_logged(self, mock_logger):
      # ... trigger warning condition ...
      mock_logger.warning.assert_called()
  ```

## Assertions

- Use `unittest` assertions: `assertEqual()`, `assertIn()`, `assertTrue()`, `assertRaises()`
- Use `numpy` for numeric comparisons: `np.testing.assert_allclose(result, expected, rtol=1e-5)`
- Check for NaN/Inf: `self.assertTrue(np.isnan(result))`, `self.assertTrue(np.isinf(result))`

## Data Type Awareness

Tests must cover both analysis modes where applicable:

- `"longitudinal"` — uses `*_frequency_diff_mean` columns, tests absolute-value analysis
- `"single"` — uses `*_frequency` columns, no absolute-value tests

## Logging

Suppress noisy logging during tests at module level:

```python
logging.getLogger().setLevel(logging.WARNING)
```

## What to Test

1. **Happy path** with representative data
2. **Edge cases**: empty DataFrames, single-sample groups, NaN values, zero coverage, identical values
3. **Both data types** (`"single"` and `"longitudinal"`) where the code branches on `data_type`
4. **Error conditions**: missing files, invalid inputs — verify graceful handling (return `None`, log warning)
5. **Statistical correctness**: verify p-values, test statistics against known results using `np.testing.assert_allclose()`

## Reference

See [tests/evolution/TEST_DOCUMENTATION.md](../../tests/evolution/TEST_DOCUMENTATION.md) for an example of thorough test documentation including mathematical formulas and boundary conditions.
