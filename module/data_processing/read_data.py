import numpy as np
import pandas as pd
import pickle


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
