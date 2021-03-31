import os, umap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
from scipy.sparse import csc_matrix
from sklearn.decomposition import PCA

#####################


def read_h5(path):
    """
    Function takes data in h5 file format and returns pandas DataFrame
    Input: (string) file pathname
    Returns: (pd.DataFrame) pandas dataframe with RNA-seq data
    """
    # Read h5 file
    h5_file = h5py.File(path)["mm10"]

    #define tuple with data, indices, and indptr
    dat_ind_ptr = (h5_file.get('data')[:],
                   h5_file.get('indices')[:], h5_file.get('indptr')[:])

    #decompress data from compressed matrix format
    csc = csc_matrix(dat_ind_ptr, h5_file.get('shape')[:])

    # return dataframe
    return pd.DataFrame(csc.toarray(),
                        columns = list(h5_file.get('barcodes')[:]),
                        index = list(h5_file.get('genes')[:]))

def main():
    files_to_plot = 5
    #initialize
    data_mat = None
    shapes = []
    for filename in os.listdir('../data/GSE115943_RAW/')[:files_to_plot]:
        #read data
        data = read_h5('../data/GSE115943_RAW/' + filename)
        #log transform and transpose data
        data = np.transpose(np.log2(data.to_numpy() + 1))
        if data_mat is None:
            data_mat = data
        else:
            data_mat = np.concatenate((data_mat, data))
        shapes.append(data.shape[0])

    # run PCA with top 30 svs
    pca = PCA(n_components=30)
    pca.fit(data_mat)
    mat_pca = pca.transform(data_mat)

    # Define UMAP
    rna_umap = umap.UMAP(n_neighbors=30, min_dist=.25)

    # Fit UMAP and extract latent vars 1-2
    embedding = pd.DataFrame(rna_umap.fit_transform(mat_pca),
                            columns = ['UMAP1','UMAP2'])
    legend_hue = [str(i) for i in range(files_to_plot) for j in range(shapes[i])]

    # Produce sns.scatterplot and use file number as color
    sns_plot = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding,
                               hue=legend_hue,
                               alpha=1, linewidth=0, s=1)
    # Adjust legend
    sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.show()
    # Save PNG
    sns_plot.figure.savefig('../figures/two_page_figure.png',
                            bbox_inches='tight', dpi=500)
main()
