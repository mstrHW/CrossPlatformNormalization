import pytest
from module.data_processing.data_generator import NoisedDataGenerator
from definitions import *
from module.data_processing.read_data import read_csv, read_genes


def test_data_generation():
    assert False


def test_using_data_generator():
    data = read_csv(illu_file, None)
    using_geos = data['GEO'].unique()
    using_genes = read_genes(best_genes_file)[:1000]
    ref_geo_name = data['GEO'].value_counts().index.values[0]
    print(ref_geo_name)
    generator = NoisedDataGenerator(data, ref_geo_name, using_geos, using_genes, batch_size=256)
