from sklearn.preprocessing import MinMaxScaler

from module.data_processing.data_processing import filter_data, normalize_by_series, apply_log, revert_log
from module.data_processing.read_data import load_test_data, load_data


class ProcessingConveyor(object):
    def __init__(self, processing_sequence):

        self.conveyor_map = {
            'filter_data': filter_data,
            'normalization': self.normalize_data,
            'apply_logarithm': self.apply_log,
        }

        self.reverse_actions_map = {
            'normalization': self.normalize_data,
            'apply_logarithm': self.revert_log,
        }

        self.processing_sequence = processing_sequence

        if 'load_test_data' in self.processing_sequence.keys():
            self.input_data, self.best_genes = load_test_data(**self.processing_sequence['load_test_data'])
        else:
            self.input_data, self.best_genes = load_data(**self.processing_sequence['load_data'])

        self.scaler = None
        self.processed_data = self.parse_sequence()

    def normalize_data(self, data, method='default'):
        _data = data.copy()
        if method == 'default':
            self.scaler = MinMaxScaler()
            self.scaler.fit(data[self.best_genes])
            _data.loc[:, self.best_genes] = self.scaler.transform(_data[self.best_genes])
        elif method == 'series':
            _data = normalize_by_series(_data, self.best_genes)
        else:
            print('Unknown normalization method')

        return _data

    def revert_normalized_data(self, data, method='default'):
        _data = data.copy()
        if method == 'default':
            _data.loc[:, self.best_genes] = self.scaler.inverse_transform(_data[self.best_genes])
        elif method == 'series':
            # _data = normalize_by_series(_data, self.best_genes) # TODO: implement method
            pass
        else:
            print('Unknown normalization method')

        return _data

    def apply_log(self, data, shift=0.):
        _data = data.copy()
        _data.loc[:, self.best_genes] = apply_log(_data[self.best_genes], shift)
        return _data

    def revert_log(self, data, shift=0.):
        _data = data.copy()
        _data.loc[:, self.best_genes] = revert_log(_data[self.best_genes], shift)
        return _data

    def parse_sequence(self):
        data = self.input_data
        items = list(self.processing_sequence.items())

        for method, params in items[1:]:
            data = self.conveyor_map[method](data, **params)
        return data

    def revert_data(self):
        data = self.input_data
        items = list(self.processing_sequence.items())

        for method, params in reversed(items[1:]):
            data = self.conveyor_map[method](data, **params)
        return data


def demo():
    processing_sequence = {
        'load_test_data': dict(
            features_count=1000,
            rows_count=None,
        ),
        'filter_data': dict(
            filtered_column='Tissue',
            using_values='Whole blood',
        ),
        'normalization': dict(
            method='series',
        ),
    }

    data = ProcessingConveyor(processing_sequence)
    print(data.processed_data[data.best_genes])


if __name__ == '__main__':
    demo()
