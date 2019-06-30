from sklearn.preprocessing import MinMaxScaler

from module.data_processing.data_processing import load_test_data, load_data, filter_data, normalize_by_series, apply_log


class processing_conveyor(object):
    def __init__(self, processing_sequence):

        self.conveyor_map = {
            'filter_data': filter_data,
            'normalization': self.normalize_data,
            'apply_logarithm': self.apply_log,
        }

        self.processing_sequence = processing_sequence

        if 'load_test_data' in self.processing_sequence.keys():
            self.input_data, self.best_genes = load_test_data(**self.processing_sequence['load_test_data'])
        else:
            self.input_data, self.best_genes = load_data(**self.processing_sequence['load_data'])

        self.scaler = None
        self.processed_data = self.parse_sequence()

    def normalize_data(self, data, method='default'):
        if method == 'default':
            self.scaler = MinMaxScaler()
            self.scaler.fit(data[self.best_genes])
            data.loc[:, self.best_genes] = self.scaler.transform(data[self.best_genes])
        elif method == 'series':
            data = normalize_by_series(data, self.best_genes)
        else:
            print('Unknown normalization method')

        return data

    def apply_log(self, data, shift=0.):
        data.loc[:, self.best_genes] = apply_log(data[self.best_genes], shift)
        return data

    def parse_sequence(self):
        data = self.input_data
        items = list(self.processing_sequence.items())
        for method, params in items[1:]:
            data = self.conveyor_map[method](data, **params)
        return data


def demo():
    # data_params = dict(
    #     features_count=1000,
    #     rows_count=None,
    #     filtered_column='Tissue',
    #     using_values='Whole blood',
    #     target_column='Age',
    #     normalize=True,
    #     use_generator=False,
    #     noising_method=None,
    #     batch_size=128,
    # )

    processing_sequence = {
        'load_data': dict(
            features_count=1000,
            rows_count=None,
        ),
        'filter_data': dict(
            filtered_column='Tissue',
            using_values='Whole blood',
        ),
        'apply_logarithm': dict(
            shift=3,
        ),
        'normalization': dict(
            method='series',
        ),
    }

    data = processing_conveyor(processing_sequence)
    print(data.processed_data[data.best_genes])


if __name__ == '__main__':
    demo()
