import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
from constants import *
from models import top_models


def single_classifier(name, results):
    type_list = list()
    accuracy_list = list()

    model = results[name]
    for type in ['Overall', 'Non-event', 'Left', 'Right']:
        type_list.extend([type for _ in range(len(model[type]))])
        accuracy_list.extend(model[type])

    long_dataframe = pd.DataFrame({
        'Type': type_list,
        'Accuracy': accuracy_list
    })

    sns.boxplot(long_dataframe,
                x='Accuracy',
                y='Type',
                palette=sns.color_palette('muted'),
                linewidth=PLOT_LINE_WIDTH)
    plt.title(f'CV Performance of {name}')
    plt.grid(False)
    plt.show()


def compare_classifiers(results_dict, model_names=None, accuracy_types=None):
    model_list = list()
    type_list = list()
    accuracy_list = list()

    if model_names == None:
        model_names = results_dict.keys()

    if accuracy_types == None:
        accuracy_types = ['Overall']

    for name in model_names:
        accuracies = results_dict[name]
        name_shorter = name[:len(name.split(':')[0])]

        for accuracy_type in accuracy_types:
            model_list.extend(
                [name_shorter for _ in range(len(accuracies[accuracy_type]))])
            type_list.extend(
                [accuracy_type for _ in range(len(accuracies[accuracy_type]))])
            accuracy_list.extend(accuracies[accuracy_type])

    combined_dataframe = pd.DataFrame({
        'Model': model_list,
        'Type': type_list,
        'Accuracy': accuracy_list
    })

    sns.boxplot(data=combined_dataframe, x='Accuracy', y='Model',
                hue='Type', palette=sns.color_palette('muted'),
                linewidth=PLOT_LINE_WIDTH)
    plt.title(f'CV Performance of Multiple Classifiers')
    plt.ylabel('Accuracy')
    plt.grid(False)
    plt.show()


def compare_classifier_methods(model_performance,
                               title='Comparing Accuracy vs. Execution Time Between Classifiers'):
    relevant_models = {k: v for k, v in model_performance.items() if k[1] == 1}
    model_names = [model[0] for model in relevant_models]
    model_types = [name.split(' #')[0] for name in model_names]
    accuracies = [model['accuracy'] for _, model in relevant_models.items()]
    times = [model['time'] for _, model in relevant_models.items()]

    data = pd.DataFrame({
        'Model': model_names,
        'Model Type': model_types,
        'Accuracy': accuracies,
        'Execution Time (seconds)': times
    })

    sns.scatterplot(data,
                    x='Execution Time (seconds)',
                    y='Accuracy',
                    hue='Model Type')
    plt.title(title)
    plt.show()


def compare_down_sample_rates(model_performance,
                              title='Comparing Accuracy vs. Execution Time Between Down Sample Rates'):
    relevant_models = {k: v for k,
                       v in model_performance.items() if k[0] in top_models}
    model_names = [model[0] for model in relevant_models]
    down_sample_rates = [model[1] for model in relevant_models]
    accuracies = [model['accuracy'] for _, model in relevant_models.items()]
    times = [model['time'] for _, model in relevant_models.items()]

    data = pd.DataFrame({
        'Model': model_names,
        'Down Sample Rate': down_sample_rates,
        'Accuracy': accuracies,
        'Execution Time (seconds)': times
    })

    sns.scatterplot(data,
                    x='Execution Time (seconds)',
                    y='Accuracy',
                    hue='Down Sample Rate',
                    palette=sns.color_palette())
    plt.title(title)
    plt.show()
