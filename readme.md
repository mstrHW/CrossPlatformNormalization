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
|   └── notebooks
|       ├── analyze_data.ipynb
|       ├── data_processing.ipynb
|       └── collect_results.ipynb
└── module                          # Base folder for implemented methods and models 
    ├── data_processing     #
    |   ├── data_processing.py
    |   ├── nosing_methods.py
    |   ├── distance_noise_generation
    |   └── processing_conveyor   
    ├── models              #
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
├── experiment_1
├── experiment_2
    ├── model_1
    └── model_2
        ├── cv_results.json
        ├── experiment_meta_parameters.json
        ├── log.log
        └── trained_models
            ├── cv_0
            └── cv_1
                ├── 0_fold
                └── 1_fold
                    ├── loss_history
                    ├── model
                    ├── model.checkpoint
                    └── tensorboard_log
```

## Data processing

### Normalization

### Logarithm

### Noising

### Processing conveyor

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

### Generating noise

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

## Models

1. MLP
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
2. DAE
3. DAE with predictor

## Experiments

### Predict age with mlp

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

### Genes normalization with dae

* *Experiment directory*: /genes_normalization/dae/
* *Best model*: ---
* *Main script*: main_scripts/genes_normalization.py
    
script_parameters:

| parameter | type | default value | description |
| --- | --- | --- | --- |
| experiment_dir | str | --- | path to directory for current experiment trained models and results |
| cv_results_file_name | str | cv_results.json| file name of search parameters results |
| search_method | str | random | choose 'random' or 'grid' search |
| n_iters | int | 100 | number of search iterations for random search |
| cuda_device_number | str | 0 | number of gpu for execute tensorflow |

### Predict age with mlp (+ logarithm on data)

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

### Genes normalization and predict age using dae with predictor
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