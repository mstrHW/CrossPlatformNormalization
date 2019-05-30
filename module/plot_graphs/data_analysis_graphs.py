import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from definitions import *


def plot_tsne_seaborn(tsne, target_column, title=None):
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne[:, 0]
    df_subset['tsne-2d-two'] = tsne[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=target_column,
        palette=sns.color_palette("hls", target_column.nunique()),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    if title is not None:
        image_name = os.path.join(IMAGES_DIR, '{}.png'.format(title))
        plt.savefig(image_name)

        
def plot_tsne_matplotlib(tsne, target_column, title=None):
    le = LabelEncoder()
    labels = le.fit_transform(target_column)
    
    plt.figure(figsize=(16, 10))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels)

    if title is not None:
        image_name = os.path.join(IMAGES_DIR, '{}.png'.format(title))
        plt.savefig(image_name)


def plot_kde(x, title):
    # x = data[data['GEO'] == geo_name][gene_name]

    kde = kde_fit(x, 1)
    x_grid = np.linspace(x.min(), x.max(), 1000)

    np.random.seed(0)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    pdf = sample_from_kde(kde, x_grid)
    ax.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    # title = 'kde_{}_{}'.format(geo_name, gene_name)
    ax.set_title(title)

    image_name = os.path.join(IMAGES_DIR, '{}.png'.format(title))
    plt.savefig(image_name)


def plot_barplot(x, title, period=None):
    # x = data[data['GEO'] == geo_name][gene_name]

    if period is None:
        period = (x.max() - x.min()) / 20

    bins = list(np.arange(x.min(), x.max(), period))
    out = pd.cut(x, bins=bins)
    ax = out.value_counts(sort=False).plot.bar(color="b", figsize=(32, 18))
    # title = 'barplot_{}_{}'.format(geo_name, gene_name)
    ax.set_title(title)

    image_name = os.path.join(IMAGES_DIR, '{}.png'.format(title))
    plt.savefig(image_name)


def plot_bar(x, title, fig_size=(16, 9)):
    pd.value_counts(x).plot.bar(rot=0, color="b", figsize=fig_size)

    image_name = os.path.join(IMAGES_DIR, '{}.png'.format(title))
    plt.savefig(image_name)


def plot_age_histogram(column, title, period=5):
    column = column[column.apply(lambda x: x.isnumeric())]
    column = column.astype(float)
    bins = list(np.arange(column.min(), column.max(), period))
    out = pd.cut(column, bins=bins, include_lowest=True)
    out.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(16, 9))

    image_name = os.path.join(IMAGES_DIR, '{}.png'.format(title))
    plt.savefig(image_name)
