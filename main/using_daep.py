from sklearn.preprocessing import MinMaxScaler

from module.data_processing.data_processing import *
from module.models.additional_dae_keras import DAEwithPredictor
from module.models.mlp import MLP

# logging.basicConfig(level=logging.DEBUG, filename=r'log.log')
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


input_data, best_genes = load_data(data_params['features_count'], data_params['rows_count'])
input_data = filter_data(input_data, data_params['filtered_column'], data_params['using_values'])

train_data, test_data = get_train_test(input_data)
train_X, train_y = get_X_y(train_data, using_genes=best_genes, target_column=data_params['target_column'])
test_X, test_y = get_X_y(test_data, using_genes=best_genes, target_column=data_params['target_column'])

logging.debug('Normalize data')
scaler = None
if data_params['normalize']:
    scaler = MinMaxScaler()
    scaler.fit(input_data[best_genes])

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

# train_y = np.array([train_y]).T
# train_y = np.concatenate((train_X, train_y), axis=1)
#
# test_y = np.array([test_y]).T
# test_y = np.concatenate((test_X, test_y), axis=1)

daep_train_y = [train_X, train_y]
daep_test_y = [test_X, test_y]

experiment_dir_name = 'genes_normalization/daep'
model_dir = path_join(MODELS_DIR, experiment_dir_name)
make_dirs(model_dir)

loss_history_file = os.path.join(model_dir, 'loss_history')
model_checkpoint_file = os.path.join(model_dir, 'model.checkpoint')
tensorboard_dir = os.path.join(model_dir, 'tensorboard_log')
model_file = os.path.join(model_dir, 'model')

model = DAEwithPredictor(
    data_params['features_count'],
    layers=[
        (1000, 384, 64),
        (384, 1000)
    ],
    activation='elu',
    regularizer_name='l1_l2',
    regularizer_param=1e-3,
    epochs_count=2000,
    learning_rate=0.0001,
    loss='mae',
    patience=200,
    optimizer_name='adam',
)

if os.path.exists(model_file):
    model.load_model(model_file)
else:
    model.fit(
        (train_X, daep_train_y),
        (test_X, daep_test_y),
        use_early_stopping=True,
        loss_history_file_name=loss_history_file,
        model_checkpoint_file_name=model_checkpoint_file,
        tensorboard_log_dir=tensorboard_dir,
    )

    model.save_model(model_file)

train_score = model.score(
    (train_X, daep_train_y),
    ['r2', 'mae'],
    scaler,
)

print(train_score)

test_score = model.score(
    (test_X, daep_test_y),
    ['r2', 'mae'],
    scaler,
)

print(test_score)

decoded_train_X = model.predict(train_X)[0]
decoded_test_X = model.predict(test_X)[0]

model_params = dict(
    layers=(1500, 800, 700, 500, 128, 1),
    activation='elu',
    drop_rate=0.5,
    regularizer_name='l1_l2',
    regularizer_param=1e-3,
    epochs_count=2000,
    loss='mae',
    batch_size=data_params['batch_size'],
    patience=200,
    optimizer_name='adam',
)

model = MLP(features_count=data_params['features_count'], **model_params)

learning_params = dict(
    use_early_stopping=True,
    loss_history_file_name=None,
    model_checkpoint_file_name=None,
)

logging.debug('Fit model')
model.fit(
    (decoded_train_X, train_y),
    (decoded_test_X, test_y),
    **learning_params,
)

train_score = model.score_sklearn(
    (train_X, train_y),
    ['r2', 'mae'],
)

print(train_score)

test_score = model.score_sklearn(
    (test_X, test_y),
    ['r2', 'mae'],
)

print(test_score)