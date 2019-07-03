import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from tqdm import tqdm_notebook
import scipy


def calculate_tsne(data):
    tsne_results = TSNE(n_components=2).fit_transform(data)
    return tsne_results


def check_normal_distribution(data, geo_names, gene_names):
    answer = []
    for geo in tqdm_notebook(geo_names):
        for gene in tqdm_notebook(gene_names):
            x = data[data['GEO'] == geo][gene]
            if len(x) < 3:
                print('{}_{}'.format(geo, gene))
            else:
                w, p = scipy.stats.shapiro(x)
                answer.append([geo, gene, w, p])
    return answer


def kde_fit(x, bandwidth=0.2, **kwargs):
    kde = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde.fit(x[:, np.newaxis])
    return kde


def sample_from_kde(kde, x_grid):
    log_pdf = kde.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)
