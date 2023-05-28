import pandas as pd
from performance_eval import test_time
from constants import *
from models import models, top_models, top_features
from feature_selection import selection_methods


def load_results(file_path):
    return pd.read_csv(file_path)


def write_results(file_path, results):
    results.to_csv(file_path, index=False)


def write_time_results(model_function,
                       model_name,
                       down_sample_rate,
                       training_data,
                       feature_selection='none',
                       num_repeats=50):

    try:
        results = load_results(f'{TIMES_DIRECTORY}/times.csv')
        results = results[[
            'model',
            'down_sample_rate',
            'feature_selection',
            'time'
        ]]
    except:
        results = pd.DataFrame(columns=[
            'model',
            'down_sample_rate',
            'feature_selection',
            'time'
        ])

    new_time = test_time(
        training_data,
        model_function(),
        down_sample_rate,
        num_repeats=num_repeats)

    new_data = pd.DataFrame({
        'model': [model_name],
        'down_sample_rate': [down_sample_rate],
        'feature_selection': [feature_selection],
        'time': [new_time]
    })

    results = pd.concat([results, new_data])

    write_results(
        f'{TIMES_DIRECTORY}/times.csv',
        results)

    return results


if __name__ == '__main__':
    training_data = pd.read_csv(f'{TRAINING_DIRECTORY}/{1}.csv')
    for name, method in models.items():
        write_time_results(method,
                           name,
                           1,
                           training_data)

    for down_sample_rate in DOWN_SAMPLE_RATES:
        training_data = pd.read_csv(
            f'{TRAINING_DIRECTORY}/{down_sample_rate}.csv')
        if down_sample_rate == 1:
            continue

        for name, method in top_models.items():
            write_time_results(method,
                               name,
                               down_sample_rate,
                               training_data)

    training_data = pd.read_csv(f'{TRAINING_DIRECTORY}/{DOWN_SAMPLE_RATE}.csv')
    for selection_name, _ in selection_methods.items():
        for name, method in top_models.items():
            write_time_results(method,
                               name,
                               DOWN_SAMPLE_RATE,
                               training_data,
                               top_features[name])
