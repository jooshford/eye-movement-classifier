import pandas as pd
from filter_training_windows import filter_training_windows
from write_training_data import write_training_data
from write_accuracy_results import write_accuracy_results
from write_time_results import write_time_results
from models import models, top_models
import plots
from constants import *


def get_model_performance(results, times):
    grouped_results = results.groupby(['model', 'down_sample_rate']).mean()
    grouped_times = times.groupby(['model', 'down_sample_rate']).mean()

    return {model: {
        'accuracy': grouped_results['accuracy_overall'][model],
        'time': grouped_times['time'][model]
    } for model in grouped_results.index}


if __name__ == '__main__':
    results = pd.read_csv(f'{RESULTS_DIRECTORY}/results.csv')
    times = pd.read_csv(f'{TIMES_DIRECTORY}/times.csv')

    new_models = ['MLP #4: [20, 10] (tanh)', 'Naive Bayes #2: Bernoulli']
    new_functions = [models[model] for model in new_models]
    for down_sample_rate in DOWN_SAMPLE_RATES:
        if down_sample_rate == 1:
            continue

        training_data = pd.read_csv(
            f'{TRAINING_DIRECTORY}/{down_sample_rate}.csv')
        write_accuracy_results(new_functions,
                               new_models,
                               down_sample_rate,
                               training_data,
                               num_repeats=10)

        for i in range(len(new_models)):
            write_time_results(new_functions[i], new_models[i],
                               down_sample_rate,
                               training_data)

    model_performance = get_model_performance(results, times)

    plots.compare_classifier_methods(model_performance)
    plots.compare_down_sample_rates(model_performance)
