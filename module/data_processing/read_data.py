import numpy as np
import pandas as pd
import pickle
import logging

from definitions import *


def read_csv(file_name, nrows=100):
    with open(file_name, newline='') as csvfile:
        data = pd.read_csv(csvfile, nrows=nrows, low_memory=False)
        return data


def read_genes(file_name):  
    with open(file_name,'rb') as f:
        x = pickle.load(f)
        return x


def read_landmarks(file_name):
    with open(file_name, 'r') as file:
        landmarks = file.read().splitlines()
    return np.array(landmarks)


def load_test_data(features_count, rows_count=None):
    logging.debug('Read files')
    data = read_csv(test_file, rows_count)
    best_genes = read_genes(best_genes_file)[:features_count]

    return data, best_genes


def create_test_data_file(data):
    test_df = data.groupby('GEO').apply(lambda x: x[:50])
    test_df.to_csv(test_file, index=False)


def load_data(features_count, rows_count=None):
    logging.debug('Read files')
    data = read_csv(illu_file, rows_count)
    best_genes = read_genes(best_genes_file)[:features_count]

    return data, best_genes