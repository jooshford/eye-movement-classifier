import pandas as pd
from constants import *


def get_model_performance():
    results = pd.read_csv(f'{RESULTS_DIRECTORY}/results.csv')
    times = pd.read_csv(f'{TIMES_DIRECTORY}/times.csv')

    grouped_results = results.groupby(
        ['model', 'down_sample_rate', 'feature_selection']).mean()
    grouped_times = times.groupby(
        ['model', 'down_sample_rate', 'feature_selection']).mean()

    return {model: {
        'accuracy': grouped_results['accuracy_overall'][model],
        'time': grouped_times['time'][model],
        'false_positives': grouped_results['false_positives'][model],
        'misclassifications': grouped_results['misclassifications'][model],
        'num_events': grouped_results['num_events'][model]
    } for model in grouped_results.index}
