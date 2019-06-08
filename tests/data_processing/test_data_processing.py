import pytest
from definitions import *
from module.data_processing.read_data import read_csv, read_genes
from module.data_processing.data_processing import load_data, get_train_test, make_shifted_data_generator


import unittest


class TestDataProcessing(unittest.TestCase):

  def setUp(self) -> None:
      features_count = 1000
      tissue = 'Whole blood'
      rows_count = 1000

      data, best_genes = load_data(features_count, tissue, rows_count)

      self.data = data
      self.best_genes = best_genes

      assert data.shape[0] == rows_count
      assert len(best_genes) == features_count
      assert data['Tissue'].unique() == tissue


