import os
import sys
import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = 'D:/Datasets/InsilicoMedicine'
sys.path.append(ROOT_DIR)

DATA_DIR = '/home/aonishchuk/data'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

illu_file = os.path.join(DATA_DIR, 'illu_rawnorm.csv')
genes_file = os.path.join(DATA_DIR, 'illu_genes.pkl')
landmarks_file = os.path.join(DATA_DIR, 'landmarks.txt')
kkochetov_model_params_file = os.path.join(DATA_DIR, 'archs.json')
best_genes_file = os.path.join(DATA_DIR, 'illu_dfs.pkl')

np_seed = 5531
sklearn_seed = 23

np.random.seed(np_seed)
tf.set_random_seed(np_seed)


def get_inner_dirs(path):
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            yield file


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def path_join(left, right):
    return os.path.join(left, right)


make_dirs(IMAGES_DIR)
make_dirs(MODELS_DIR)
