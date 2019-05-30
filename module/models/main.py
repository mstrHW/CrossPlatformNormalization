import csv
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
import sys
sys.path.append("../") # go to parent dir

from read_data import *
from plot_graphs import *
import json
from sklearn.preprocessing import normalize, MinMaxScaler, minmax_scale
import keras


ROOT_DIR = '/home/aonishchuk/'
DATA_DIR = os.path.join(ROOT_DIR, 'data')

illu_file = os.path.join(DATA_DIR, 'illu_rawnorm.csv')
genes_file = os.path.join(DATA_DIR, 'illu_genes.pkl')
landmarks_file = os.path.join(DATA_DIR, 'landmarks.txt')
kkochetov_model_params_file = os.path.join(DATA_DIR, 'archs.json')
best_genes_file = os.path.join(DATA_DIR, 'illu_dfs.pkl')


data = read_csv(illu_file, None)
gene_names = get_genes(genes_file)
best_genes = get_genes(best_genes_file)
print(best_genes.shape)


print(data.shape)
data_float = data.copy()
data_float = data_float[pd.to_numeric(data_float['Age'], errors='coerce').notnull()]
data_float['Age'] = data_float['Age'].fillna(data_float['Age'].mean())
print(data_float.shape)

train_mask = data_float['Training'] == 'Y'
train_data = data_float[train_mask]
test_data = data_float[~train_mask]

# from sklearn.model_selection import train_test_split
# train_data, test_data = train_test_split(data_float, test_size=0.3, random_state=42)
print(train_data.shape)



scaler = MinMaxScaler()

using_genes = best_genes[:1000]
normalized_data = data_float.copy()
normalized_data[using_genes] = scaler.fit_transform(normalized_data[using_genes])
# normalized_data[using_genes] = normalized_data.groupby('GEO')[using_genes].transform(lambda x: minmax_scale(x))
normalized_train_data = normalized_data[train_mask]
normalized_test_data = normalized_data[~train_mask]


from mlp import create_model

train_X = normalized_train_data[using_genes].astype('float')
train_y = normalized_train_data['Age'].astype('float')

test_X = normalized_test_data[using_genes].astype('float')
test_y = normalized_test_data['Age'].astype('float')

features_count = train_data[using_genes].values.shape[1]
epochs_count = 700
batch_size = 128
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=10, patience=100, restore_best_weights=True)
callbacks_list = [es]
    
model = create_model(features_count)
model.fit(train_X, train_y, epochs=epochs_count, batch_size=batch_size, validation_data=(test_X, test_y), callbacks=callbacks_list)
print_score(model, normalized_train_data, normalized_test_data)
