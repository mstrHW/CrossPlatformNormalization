import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.append(ROOT_DIR)

DATA_DIR = '/home/aonishchuk/data'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

illu_file = os.path.join(DATA_DIR, 'illu_rawnorm.csv')
genes_file = os.path.join(DATA_DIR, 'illu_genes.pkl')
landmarks_file = os.path.join(DATA_DIR, 'landmarks.txt')
kkochetov_model_params_file = os.path.join(DATA_DIR, 'archs.json')
best_genes_file = os.path.join(DATA_DIR, 'illu_dfs.pkl')
