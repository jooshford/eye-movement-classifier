import sys
import pandas as pd
from constants import *
from models import models, top_models
from performance_eval import run_n_times


accuracy_measures = [
    'model',
    'down_sample_rate',
    'feature_selection',
    'accuracy_overall',
    'accuracy_N',
    'accuracy_L',
    'accuracy_R',
    'accuracy_B',
    'false_positives',
    'misclassifications',
    'num_events'
]


def load_results(file_path):
    return pd.read_csv(file_path)


def write_results(file_path, results):
    results.to_csv(file_path, index=False)


def write_accuracy_results(model_functions,
                           model_names,
                           down_sample_rate,
                           training_data,
                           feature_selection='none',
                           num_repeats=50):
    try:
        results = load_results(
            f'{RESULTS_DIRECTORY}/results.csv')
        results = results[accuracy_measures]
    except:
        results = pd.DataFrame(columns=accuracy_measures)

    performances = run_n_times(
        model_functions,
        training_data,
        n=num_repeats
    )

    for i in range(len(performances)):
        new_data = pd.DataFrame({
            'model': [model_names[i] for _ in range(len(performances[i]))],
            'down_sample_rate': [down_sample_rate for _ in range(len(performances[i]))],
            'feature_selection': [feature_selection for _ in range(len(performances[i]))],
            'accuracy_overall': [x.accuracy() for x in performances[i]],
            'accuracy_N': [x.accuracy('N') for x in performances[i]],
            'accuracy_L': [x.accuracy('L') for x in performances[i]],
            'accuracy_R': [x.accuracy('R') for x in performances[i]],
            'accuracy_B': [x.accuracy('B') for x in performances[i]],
            'false_positives': [x.count_false_positives() for x in performances[i]],
            'misclassifications': [x.count_misclassified_events() for x in performances[i]],
            'num_events': [x.count_events() for x in performances[i]]
        })

        results = pd.concat([results, new_data])

    write_results(
        f'{RESULTS_DIRECTORY}/results.csv',
        results)

    return results
