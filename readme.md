# CrossPlatformNormalization


## Project structure

```
.
├── definitions.py                  # Defines key folders, files and variables for project
├── main_scripts                    # Scripts for executing experiments and getting results
|   ├── genes_normalization.py
|   ├── predict_age.py
|   ├── predict_age_after_dae.py
|   ├── predict_age_log_data.py
|   ├── using_daep.py
|   ├── utils.py
|   └── notebooks                   # Jupyter notebooks were used for data analysis and experiments results visualization
|       ├── analyze_data.ipynb
|       ├── data_processing.ipynb
|       └── collect_results.ipynb
└── module                          # Base folder for implemented methods and models 
    ├── data_processing             # Defines methods of data loading, preprocessing and noising
    |   ├── data_processing.py
    |   ├── nosing_methods.py
    |   ├── distance_noise_generation
    |   └── processing_conveyor   
    ├── models                      # Defines classes for each used in experiments models
    |   ├── base_model.py
    |   ├── dae.py
    |   ├── dae_with_predictor.py
    |   ├── mlp.py
    |   └── utils
    |       ├── activations.py
    |       ├── callbacks.py
    |       ├── metrics.py
    |       ├── optimizers.py
    |       ├── regularizers.py
    |       ├── grid_search.py
    |       └── utils.py
    └── plot_graphs

```

## Experiment structure

```
.
├── experiment_1                                # folder for first experiment
├── experiment_2                                # folder for second experiment
    ├── model_1                                 # folder for first model
    └── model_2                                 # folder for second model
        ├── cv_results.json                     # contains parameters, scores and directory of models gived by search parameters method
        ├── experiment_meta_parameters.json     # numpy and sklearn seed values
        ├── data_parameters.json                # preprocessing sequence for data
        ├── log.log                             # contains key stages of search method executing
        └── trained_models
            ├── cv_0                            # folder for first set of parameters
            └── cv_1                            # folder for second set of parameters
            ├── 0_fold                          # first set of cross-validation results
            └── 1_fold
                ├── loss_history                # contains train and test scores collected during training
                ├── model                       # saved model file
                ├── model.checkpoint            # last checkpoint file
                └── tensorboard_log             # contains parameters and scores collected during training
```

## Data processing (code_path: module/data_processing)

For data processing was used following methods:
1. Normalization (MinMaxScale from sklearn)
2. Series normalization (MinMaxScale for each group of GEO in data separately)  (code_file: data_processing.py, method: series_normalization)
3. Logarithm (code_file: data_processing.py, method: apply_log)

### Processing conveyor (code_file: module/data_processing/processing_conveyor)
For more convenient usage and documentation of methods ProcessingConveyor class was implemented.
#### Usage

```python
from module.data_processing.processing_conveyor import ProcessingConveyor

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

processing_conveyor = ProcessingConveyor(processing_sequence)
best_genes = processing_conveyor.best_genes
processed__data = processing_conveyor.processed_data
```

### Generating noise (code_path: module/data_processing)
For data noising was used following methods:
    1. Gaussian noise (code_file: noising_methods.py, method: gaussian_noise)
    2. Distance noise: use distance of distributions (code_file: distance_noise_generation.py, class: DistanceNoiseGenerator)

#### Usage

```python
from module.data_processing.distance_noise_generation import DistanceNoiseGenerator
from module.data_processing.data_processing import get_batches

best_genes = processing_conveyor.best_genes
processed__data = processing_conveyor.processed_data

ref_batch_name = processed__data['GEO'].value_counts().keys()[0]
ref_batch_mask = processed__data['GEO'] == ref_batch_name

ref_batch = processed__data[ref_batch_mask]
corrupt_data = processed__data[~ref_batch_mask]

noise_probability = 0.5
batch_size = 128

train_noised_generator = DistanceNoiseGenerator(
    ref_batch,
    corrupt_data,
    best_genes,
    'train',
    noise_probability,
)

for batch in get_batches(ref_batch, batch_size):
    corrupt_X = train_noised_generator.data_generation(batch[best_genes])
    y = batch[best_genes]
```

## Models (code_path: module/models)

1. MLP (code_file: mlp.py)
    #### Usage
    
    ```python
    from module.models.mlp import MLP
    
    model_params = dict(
        layers=(1500, 800, 700, 500, 128, 1),
        activation='elu',
        drop_rate=0.5,
        regularizer_name='l1_l2',
        regularizer_param=1e-3,
        epochs_count=2000,
        loss='mae',
        patience=200,
        optimizer_name='adam',
    )
    
    
    model = MLP(
        features_count=1000,
        **model_params,
    )
    
    
    model.fit(train_generator)
    model.score(test_generator, ['mae', 'r2'])
    
    ```
2. DAE (code_file: dae.py)
3. DAE with predictor (code_file: dae_with_predictor.py)

## Experiments

### Predict age with mlp (code_file: main_scripts/predict_age.py)
```python
processing_sequence = {
    'load_data': dict(
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
```

Parameters space:
```python
layers = [
    (256, 128, 64, 1),
    (512, 256, 128, 1),
    (512, 384, 256, 128, 1),
    (768, 512, 384, 192, 1),
    (1024, 768, 512, 384, 128, 1),
    (1536, 1024, 768, 384, 192, 1),
]
activation = ['elu', 'lrelu', 'prelu']
dropout_rate = [0.25, 0.5, 0.75]
regularization_param = [10 ** -i for i in range(3, 7)]
epochs_count = 2000,
loss = 'mae',
optimizer = ['adam', 'rmsprop']
```

* *Experiment directory* : /predict_age/mlp/
* *Best model* : cv_19
* *Main script* : main_scripts/predict_age.py
    
script_parameters:

| parameter | type | default value | description |
| --- | --- | --- | --- |
| experiment_dir | str | --- | path to directory for current experiment trained models and results |
| cv_results_file_name | str | cv_results.json| file name of search parameters results |
| search_method | str | random | choose 'random' or 'grid' search |
| n_iters | int | 100 | number of search iterations for random search |
| cuda_device_number | str | 0 | number of gpu for execute tensorflow |

### Genes normalization with dae (code_file: main_scripts/genes_normalization.py)
Data preprocessing parameters:
```python
processing_sequence = {
    'load_data': dict(
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
```
and noising method was distance noise with 50% probability of noising genes

Parameters space:
```python
layers = [
    (
        (1000, 512, 256, 128, 64),
        (128, 256, 512, 1000)
    ),
    (
        (1000, 512, 256, 64),
        (256, 512, 1000)
    ),
    (
        (1000, 384, 64),
        (384, 1000)
    ),
]

activation = ['elu', 'lrelu', 'prelu']
regularization_param = [10 ** -i for i in range(3, 7)]
epochs_count = 2000,
loss = 'mae',
optimizer = ['adam', 'rmsprop']
```

* *Experiment directory*: /genes_normalization/dae/
* *Best model*: trained_models/cv_56
* *Main script*: main_scripts/genes_normalization.py
    
script_parameters:

| parameter | type | default value | description |
| --- | --- | --- | --- |
| experiment_dir | str | --- | path to directory for current experiment trained models and results |
| cv_results_file_name | str | cv_results.json| file name of search parameters results |
| search_method | str | random | choose 'random' or 'grid' search |
| n_iters | int | 100 | number of search iterations for random search |
| cuda_device_number | str | 0 | number of gpu for execute tensorflow |

### Predict age with mlp (+ logarithm on data) (code_file: main_scripts/predict_age_log_data.py)
Data preprocessing parameters:

```python
processing_sequence = {
    'load_test_data': dict(
        features_count=1000,
        rows_count=None,
    ),
    'filter_data': dict(
        filtered_column='Tissue',
        using_values='Whole blood',
    ),
    'apply_logarithm': dict(
        shift=3.,
    ),
    'normalization': dict(
        method='series',
    ),
}
```

* *Experiment directory*: /predict_age_log_data/
* *Best model*: --- (search parameters method was not used)
* *Main script*: main_scripts/predict_age_log_data.py
    
script_parameters:

| parameter | type | default value | description |
| --- | --- | --- | --- |
| experiment_dir | str | --- | path to directory for current experiment trained models and results |
| mlp_models_dir | str | --- | path to directory of predicting age with mlp experiment |
| results_file_name | str | results.json| name of file with score results |
| cuda_device_number | str | 0 | number of gpu for execute tensorflow |

### Genes normalization and predict age using dae with predictor  (code_file: main_scripts/using_daep.py)
#### Description:
Genes normalization and predict age using dae with predictor on latent layer. Data was preprocessed using following parameters:
```python
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
```
and noising method was distance noise with 50% probability of noising genes

* *Experiment directory*: /genes_normalization/daep
* *Best model*: ---
* *Main script*: main_scripts/using_daep.py
    
script_parameters:

| parameter | type | default value | description |
| --- | --- | --- | --- |
| experiment_dir | str | --- | path to directory for current experiment trained models and results |
| results_file_name | str | results.json| name of file with score results |
| cuda_device_number | str | 0 | number of gpu for execute tensorflow |