import sys
sys.path.append("../")
import logging

from definitions import *

logging.basicConfig(level=logging.DEBUG)
logging.debug('Read data')

data_params = dict(
    features_count=1000,
    rows_count=None,
    filtered_column='Tissue',
    using_values='Whole blood',
    target_column='Age',
    normalize=True,
    use_generator=False,
    noising_method=None,
    batch_size=128,
)

from module.data_processing.data_processing import load_data, filter_data, get_train_test
from sklearn.preprocessing import MinMaxScaler


input_data, best_genes = load_data(data_params['features_count'], data_params['rows_count'])
input_data = filter_data(input_data, data_params['filtered_column'], data_params['using_values'])


def preprocess_data(data, best_genes):
    train_data, test_data = get_train_test(data)

    ref_batch_name = 'GSE33828'
    normalized_ref_data = train_data[train_data['GEO'] == ref_batch_name].copy()
    normalized_train_data = train_data[train_data['GEO'] != ref_batch_name].copy()
    normalized_test_data = test_data.copy()

    def shift_with_log(X):
        return np.log(X + 3)

    scaler = None
    if data_params['normalize']:
        scaler = MinMaxScaler(shift_with_log(data[best_genes]))
        columns = best_genes.tolist()
        columns.append('GEO')

        my_preprocessing = lambda x: scaler.transform(shift_with_log(x))
        cutted = normalized_ref_data[columns]
        normalized_ref_data[best_genes] = cutted.groupby('GEO').transform(lambda x: my_preprocessing(x))

        cutted = normalized_train_data[columns]
        normalized_train_data[best_genes] = cutted.groupby('GEO').transform(lambda x: my_preprocessing(x))

        cutted = normalized_test_data[columns]
        normalized_test_data[best_genes] = cutted.groupby('GEO').transform(lambda x: my_preprocessing(x))

    return normalized_ref_data, normalized_train_data, normalized_test_data, scaler


normalized_ref_data, normalized_train_data, normalized_test_data, scaler = preprocess_data(input_data, best_genes)


print(normalized_train_data.shape)
print(normalized_train_data[best_genes].shape)


# train_generator = NewNoisedDataGenerator(
#     normalized_ref_data,
#     normalized_train_data,
#     best_genes,
#     'test',
#     batch_size=data_params['batch_size'],
#     noising_method=data_params['noising_method'],
#     # shift_probability=0.5,
# )
#
# test_generator = NewNoisedDataGenerator(
#     normalized_ref_data,
#     normalized_test_data,
#     best_genes,
#     'test',
#     batch_size=data_params['batch_size'],
#     noising_method=None,
#     # shift_probability=0.5,
# )


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from module.models.mlp import MLP
from main.utils import load_best_model

best_mlp_model, best_model_path = load_best_model(
    MLP,
    os.path.join(MODELS_DIR, 'predict_age/mlp'),
    'cv_results.json',
)

cv_model_path = '/home/aonishchuk/projects/CPN2/test_models/learn_mlp_with_shifted_data/normalized'
make_dirs(cv_model_path)

learning_params = dict(
    use_early_stopping=True,
    loss_history_file_name=os.path.join(cv_model_path, 'loss_history'),
    model_checkpoint_file_name=os.path.join(cv_model_path, 'model.checkpoint'),
    tensorboard_log_dir=os.path.join(cv_model_path, 'tensorboard_log'),
    # generator=True,
)


best_mlp_model.fit(
    (normalized_train_data[best_genes], normalized_train_data['Age']),
    (normalized_test_data[best_genes], normalized_test_data['Age']),
    **learning_params
)
#
# best_mlp_model.fit_generator(
#     train_generator,
#     (normalized_test_data[best_genes], normalized_test_data['Age']),
#     **learning_params
# )


best_mlp_model.save_model(os.path.join(cv_model_path, 'model'))

train_pred = best_mlp_model.predict(normalized_train_data[best_genes])
print(train_pred.shape)
print(normalized_train_data['Age'].shape)
test_pred = best_mlp_model.predict(normalized_test_data[best_genes])
#
# if data_params['normalize']:
#     train_pred = scaler.inverse_transform(train_pred)
#     test_pred = scaler.inverse_transform(test_pred)


from sklearn.metrics import mean_absolute_error, r2_score

columns = 'Age'
train_score = mean_absolute_error(normalized_train_data[columns], train_pred), r2_score(normalized_train_data[columns], train_pred)
test_score = mean_absolute_error(normalized_test_data[columns], test_pred), r2_score(normalized_test_data[columns], test_pred)

write_message = dict(
    train_results=train_score,
    test_results=test_score,
)


import logging
logging.basicConfig(level=logging.DEBUG, filename=r'log.log')

results_file = os.path.join(cv_model_path, 'results')

import json
with open(results_file, 'w') as file:
    json.dump(write_message, file)
    logging.info('overwrite model parameters file ({})'.format(results_file))
print(train_score, test_score)