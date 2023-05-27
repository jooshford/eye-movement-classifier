import pandas as pd
from performance_eval import test_time
from constants import *


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
