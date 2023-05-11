import sys
import pandas as pd
from constants import *
from models import models, top_models
from performance_eval import run_n_times


def load_results(file_path):
    return pd.read_csv(file_path)


def write_results(file_path, results):
    results.to_csv(file_path)


def write_accuracy_results(model_function,
                           model_name,
                           down_sample_rate,
                           training_data,
                           feature_selection='none',
                           num_repeats=50):
    try:
        results = load_results(
            f'{RESULTS_DIRECTORY}/results.csv')
        results = results[[
            'model',
            'down_sample_rate',
            'feature_selection',
            'accuracy_overall',
            'accuracy_N',
            'accuracy_L',
            'accuracy_R']]
    except:
        results = pd.DataFrame(columns=[
            'model',
            'down_sample_rate',
            'feature_selection',
            'accuracy_overall',
            'accuracy_N'
            'accuracy_L',
            'accuracy_R'
        ])

    performances = run_n_times(
        model_function,
        training_data,
        n=num_repeats
    )

    new_data = pd.DataFrame({
        'model': [model_name for _ in range(len(performances))],
        'down_sample_rate': [down_sample_rate for _ in range(len(performances))],
        'feature_selection': [feature_selection for _ in range(len(performances))],
        'accuracy_overall': [x.accuracy() for x in performances],
        'accuracy_N': [x.accuracy('N') for x in performances],
        'accuracy_L': [x.accuracy('L') for x in performances],
        'accuracy_R': [x.accuracy('R') for x in performances]
    })

    results = pd.concat([results, new_data])

    write_results(
        f'{RESULTS_DIRECTORY}/results.csv',
        results)

    return results
