# Problem Set 4

### OVERVIEW
The repository contains code, data, and resources for performing PCA and UMAP on single-cell RNA-seq data and plotting the results. Plots show UMAP latent variables 1 and 2 and are colored by condition (file number).

### DATA
Data was generated by Schiebinger et al. as single-cell RNA-seq analysis of cells reprogrammed to iPSCs over the course of many days.

Data can be accessed on the Gene Expression Omnibus (GEO) with accession ID GSE115943. A full citation is provided below:

Schiebinger G, Shu J, Tabaka M, Cleary B et al. Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming. Cell 2019 Feb 7;176(4):928-943.e22. PMID: 30712874

### FOLDER STRUCTURE
The repository has 3 directories: code, data, figures. The code directory contains python code for generating the figures. The data directory contains the raw data from GEO with accession id GSE115943. Raw files are in hd5 format. Figures contains the figure generated by the code in png format.

### INSTALLATION
Code can be run in the terminal from the master directory by typing the following.

<code>
cd code
</code>
<code>
python3 generate_figures.py
</code>

The following dependencies are required:
- umap
- seaborn
- matplotlib.pyplot
- pandas
- numpy
- h5py
- scipy
- sklearn.decomposition
