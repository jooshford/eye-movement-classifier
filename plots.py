import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
from constants import *


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


def scatter_time_accuracy(results,
                          times_dict,
                          title='Comparing Accuracy vs. Execuation Time Between Classifiers',
                          down_sample_rate=1):
    models = results.keys()
    model_type = [model.split(' #')[0] for model in models]
    accuracies = [np.mean(model['Overall']) for model in results.values()]
    times = [times_dict[model] for model in results.keys()]

    data = pd.DataFrame({
        'Model': models,
        'Model Type': model_type,
        'Accuracy': accuracies,
        'Execution Time (seconds)': times
    })

    sns.scatterplot(data,
                    x='Execution Time (seconds)',
                    y='Accuracy',
                    hue='Model Type')
    plt.title(title)
    plt.savefig(f'figure_{down_sample_rate}.png')
    plt.show()
