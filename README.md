This repository includes scripts for population genetic analyses on large number of MAGs.

# Manual installation
NOTE: This will be made more streamlined and automated in future versions.

1. Install [`mamba`](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) using the following commdands:

    ```bash
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```
2. Install the required packages in the environment called `popgenToolkit`

    ```bash
    mamba env create -f environment.yml
    ```

3. Activate the environment

    ```bash
    mamba activate popgenToolkit
    ```

    Now you are ready to run the scripts !

    ## Example

    ```bash
    python scripts/analyze_allele_freq.py -h
    ```
    
    This will show you all the input options for the `analyze_allele_freq.py` script
