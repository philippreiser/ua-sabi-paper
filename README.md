# Uncertainty-Aware Surrogate-Based Amortized Bayesian Inference (UA-SABI)
Supplementary code.

## Installation
Create a new environment with Python 3.11. Install the required dependencies by running:
```bash
pip install -r requirements.txt
```
Install CmdStanPy by running the following commands in Python:
```python
import cmdstanpy; cmdstanpy.install_cmdstan()
```
Alternatively, follow the [installation instructions](https://mc-stan.org/install/).

## Reproducing Results for Case Studies
To reproduce the results presented in the case studies, train surrogate models and train (UA-)SABI or run MCMC by executing:
```bash
python main.py --config <path-to-config>
```
The configuration files are provided in the `configs/` directory.

### CO₂ Case Study

To reproduce the CO₂ case study, download the dataset from:

**Köppel et al. (2017)**  
*Datasets and executables of data-driven uncertainty quantification benchmark in carbon dioxide storage*  


DOI: [10.5281/zenodo.933827](https://doi.org/10.5281/zenodo.933827)

### MICP Data

**Hommel et al. (2015)**  


*A Revised Model for Microbially Induced Calcite Precipitation: Improvements and New Insights Based on Recent Experiments*  

DOI: [10.1002/2014WR016503](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014WR016503)
