import pandas as pd
from filter_training_windows import filter_training_windows
from write_training_data import write_training_data
from write_accuracy_results import write_accuracy_results
from write_time_results import write_time_results
from models import models, top_models
import plots
from constants import *


def get_model_performance(results, times):
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


if __name__ == '__main__':
    results = pd.read_csv(f'{RESULTS_DIRECTORY}/results.csv')
    times = pd.read_csv(f'{TIMES_DIRECTORY}/times.csv')

    model_performance = get_model_performance(results, times)

    plots.compare_classifier_methods(model_performance)
    plots.compare_down_sample_rates(model_performance)
    plots.compare_feature_selection(model_performance)
    plots.compare_top_methods(model_performance)
    plots.presentation_plot(model_performance)
