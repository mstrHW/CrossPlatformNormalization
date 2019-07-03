# CrossPlatformNormalization


## Project structure

```
.
├── definitions.py          # Defines key folders, files and variables for project
├── main_scripts                    # Documentation files (alternatively `doc`)
|   ├── README.md
|   ├── README.md
|   ├── README.md
|   ├── README.md
|   ├── README.md
|   ├── README.md
|   └── notebooks
|       ├── notebooks
|       ├── notebooks
|       └── notebooks
└── module # 
    ├── data_processing     #
    |   ├── notebooks
    |   ├── notebooks
    |   ├── notebooks
    |   ├── notebooks
    |   └── notebooks        
    ├── models              #
    |   ├── base_model.py
    |   ├── dae.py
    |   ├── dae_with_predictor.py
    |   ├── mlp.py
    |   └── notebooks
    └── plot_graphs

```

## Experiment structure

```
.
├── experiment_1
├── experiment_2
|   ├── model_1
|   └── model_2
|       ├── cv_results.json
|       ├── experiment_meta_parameters.json
|       ├── log.log
|       └── trained_models
|           ├── cv_0
|           └── cv_1
|               ├── 0_fold
|               └── 1_fold
|                   ├── loss_history
|                   ├── model
|                   ├── model.checkpoint
|                   └── tensorboard_log
├── models                  # Compiled files (alternatively `dist`)
├── main_scripts                    # Documentation files (alternatively `doc`)
|   ├── README.md
|   ├── README.md
|   ├── README.md
|   ├── README.md
|   ├── README.md
|   ├── README.md
|   └── notebooks
|       ├── notebooks
|       ├── notebooks
|       └── notebooks
├── src                     # Source files (alternatively `lib` or `app`)
├── test                    # Automated tests (alternatively `spec` or `tests`)
├── tools                   # Tools and utilities
├── LICENSE
└── README.md
```

## Data processing

### Normalization

### Logarithm

### Noising

### Processing conveyor

#### Usage

```python
from module.data_processing.ProcessingConveyor import ProcessingConveyor

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

data = ProcessingConveyor(processing_sequence)
best_genes = data.best_genes
processed__data = data.processed_data
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
    
    
    model.fit(train_X, train_y)
    model.score(test_X, test_y)
    
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
| experiment_dir | str | --- | description |
| cv_results_file_name | str | cv_results.json| description |
| search_method | str | random | description |
| n_iters | int | 100 | description |
| cuda_device_number | str | 0 | description |

### Genes normalization with dae

* *Experiment directory*: /genes_normalization/dae/
* *Best model*: ---
* *Main script*: main_scripts/genes_normalization.py
    
script_parameters:

| parameter | type | default value | description |
| --- | --- | --- | --- |
| experiment_dir | str | --- | description |
| cv_results_file_name | str | cv_results.json| description |
| search_method | str | random | description |
| n_iters | int | 100 | description |
| cuda_device_number | str | 0 | description |

### Predict age with mlp (+ logarithm on data)

* *Experiment directory*: /predict_age_log_data/
* *Best model*: ---
* *Main script*: main_scripts/predict_age_log_data.py
    
script_parameters:

| parameter | type | default value | description |
| --- | --- | --- | --- |
| experiment_dir | str | --- | description |
| mlp_models_dir | str | --- | description |
| results_file_name | str | results.json| description |
| cuda_device_number | str | 0 | description |

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

* *Experiment directory*: /predict_age_log_data/
* *Best model*: ---
* *Main script*: main_scripts/predict_age_log_data.py
    
script_parameters:

| parameter | type | default value | description |
| --- | --- | --- | --- |
| experiment_dir | str | --- | description |
| mlp_models_dir | str | --- | description |
| results_file_name | str | results.json| description |
| cuda_device_number | str | 0 | description |