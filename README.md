This repository includes workflow for analyzing changes in allele frequency in bacteria and analyzing their significance.

# 🖱️ Installation 🖱️

1. Install [`mamba`](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) using the following commdands:

    ```bash
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```
2. Glone the git repository and install the required packages

    ```bash
    git clone https://github.com/MoellerLabPU/popGenToolKit.git
    cd popGenToolKit
    mamba env create -f environment.yml
    ```

3. Activate the environment

    ```bash
    mamba activate popgenToolkit
    ```

Now you are ready to run the workflow !

# 🐍 Snakemake workflow 🐍 

## 📂 Relevant Files 📂 

- `/popGenToolKit/smk_workflow/snakefile`: This is the main file that defines the Snakemake workflow. It contains the rules and dependencies for the analysis.
- `/popGenToolKit/smk_workflow/config.yml`: This file contains configuration parameters for the workflow, such as input file paths and analysis settings.
- `/popGenToolKit/smk_workflow/cornell_profile/config.yaml`: Specific run parameters to be submitted to SLURM. No need to edit it it unless you know what you're doing.

## 🏃 Running the worklow 🏃

1. You only really need to edit `/popGenToolKit/smk_workflow/config.yml` to be able to run the workflow. Change the paths to the scripts, file and any input parameters.
2. Make sure that the correct environment is activate ie. `popgenToolkit` and you're in `/popGenToolKit/smk_workflow` directory.
3. To run the worklow do, `snakemake --profile cornell_profile` and let the magic happen magic happen 🪄 👨‍🔬 👩‍🔬 !