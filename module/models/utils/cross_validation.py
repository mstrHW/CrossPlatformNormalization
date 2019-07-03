from sklearn.model_selection import KFold


def choose_cross_validation(method='default'):
    cross_validation = None
    if method == 'default':
        cross_validation = k_fold_splits_generator
    elif method == 'custom':
        cross_validation = separate_labels_splits
    return cross_validation


def separate_labels_splits(train_data, cross_validation_parameters):
    geos = train_data['GEO'].value_counts().keys()
    splits_count = cross_validation_parameters['n_splits']

    if splits_count > len(geos):
        raise ValueError

    for i, val_geo_name in enumerate(geos):
        if i < splits_count:
            cv_val_mask = train_data['GEO'] == val_geo_name

            cv_train_data = train_data[~cv_val_mask]
            cv_val_data = train_data[cv_val_mask]

            yield cv_train_data, cv_val_data


def k_fold_splits_generator(train_data, cross_validation_parameters):
    k_fold = KFold(**cross_validation_parameters)

    for train_indexes, val_indexes in k_fold.split(train_data):
        cv_train_data = train_data.iloc[train_indexes]
        cv_val_data = train_data.iloc[val_indexes]

        yield cv_train_data, cv_val_data
