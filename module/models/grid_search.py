from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import os


def save_search_results(cv_results, best_params, best_score, title):
    results_df = pd.DataFrame.from_dict(cv_results)
    results_file = os.path.join(MODELS_DIR, '{}_results.csv'.format(title))
    results_df.to_csv(results_file)

    best_model_file = os.path.join(MODELS_DIR, '{}_best_model.csv'.format(title))
    fields = ['best_params', 'best_score']
    file_exists = os.path.exists(best_model_file)

    with open(best_model_file, 'a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fields)
        writer.writerow([best_params, best_score])


def log_metrics_and_params(results, model_path):
    best_model_file = os.path.join(MODELS_DIR, '{}_best_model.csv'.format(title))
    fields = ['model_path', 'results']
    file_exists = os.path.exists(best_model_file)

    with open(best_model_file, 'a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fields)
        writer.writerow([model_path, results])


def calculate_metrics(y, y_pred):
    r2_ = r2_score(y, y_pred)
    mae_ = mean_absolute_error(y, y_pred)

    metrics = {
        'r2': r2_,
        'mae': mae_,
    }

    return metrics


def get_unique_folder_name(model_directory) -> str:
    inner_dirs = os.walk(model_directory)
    return str(len([x[0] for x in inner_dirs]))


def scoring_method(clf, X, y, model_directory):
    model_parameters = clf.get_params()
    y_pred = clf.predict(X)

    results = calculate_metrics(y, y_pred)

    model_dir_name = get_unique_folder_name(model_directory)
    model_path = os.path.join(model_directory, model_dir_name)
    os.makedirs(model_path)

    log_message = dict(
        model_parameters=model_parameters,
        model_directory=model_directory,
        mode

    model_save_path = clf.save_model(model_path)
    log_metrics_and_params(results, model_save_path)
    return results['mae']


def search_parameters(model,
                      data,
                      model_parameters_space,
                      learning_parameters,
                      gs_parameters,
                      model_directory,
                      gs_results_file,
                      search_method_name='random',
                      ):

    search_method = RandomizedSearchCV
    if search_method_name == 'grid':
        search_method = GridSearchCV

    train_data, test_data = data

    grid = search_method(model, model_parameters_space, **gs_parameters)
    grid.fit(
        *train_data,
        test_data=test_data,
        scoring=lambda clf, x, y: scoring_method(clf, x, y, model_directory),
        **learning_parameters
    )

    return grid.cv_results_, grid.best_params_, grid.best_score_
