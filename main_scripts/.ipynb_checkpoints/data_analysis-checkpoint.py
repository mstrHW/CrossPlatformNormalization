from definitions import *
from module.data_processing.read_data import read_csv, read_genes
from module.data_processing.data_understanding import calculate_tsne
from module.plot_graphs.data_analysis_graphs import plot_tsne_seaborn
import logging


logging.basicConfig(level=logging.DEBUG)
logging.debug('Read files')
data = read_csv(illu_file, None)
genes = read_genes(genes_file)
best_genes = read_genes(best_genes_file)

logging.debug('Calculate tsne')
tsne = calculate_tsne(data[best_genes])
target_columns = ['GEO', 'Tissue', 'Age_group', 'Sex']

logging.debug('Plot graphs')
for column in target_columns:
    plot_tsne_seaborn(tsne, data[column], 'tsne_{}'.format(column.lower()))
