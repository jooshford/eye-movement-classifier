import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from constants import *
from models import top_models, top_features


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
    relevant_models = {
        k: v for k, v in model_performance.items() if k[1] == DOWN_SAMPLE_RATE and k[2] == top_features[k[0]]}
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
                       v in model_performance.items() if k[0] in top_models and k[2] == 'none'}
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


def compare_feature_selection(model_performance,
                              title='Comparing Accuracy vs. Execution Time vs. Feature Selection (on KNN)'):
    relevant_models = {k: v for k,
                       v in model_performance.items() if 'KNN' in k[0] and k[1] == DOWN_SAMPLE_RATE}
    model_names = [model[0] for model in relevant_models]
    method_map = {
        'RFE_LR': 'RFE (Logistic Regression)',
        'none': 'None',
        'RFE_RF': 'RFE (Random Forest)',
        'RFE_SVM': 'RFE (SVM)',
        'f_classif': 'F-Value',
        'mutual_info': 'Mutual Information'
    }
    feature_selection_methods = [method_map[model[2]]
                                 for model in relevant_models]
    accuracies = [model['accuracy'] for _, model in relevant_models.items()]
    times = [model['time'] for _, model in relevant_models.items()]

    data = pd.DataFrame({
        'Model': model_names,
        'Feature Selection Method': feature_selection_methods,
        'Accuracy': accuracies,
        'Execution Time (seconds)': times
    })

    sns.scatterplot(data,
                    x='Execution Time (seconds)',
                    y='Accuracy',
                    hue='Feature Selection Method',
                    palette=sns.color_palette())
    plt.title(title)
    plt.show()


def compare_top_methods(model_performance,
                        title='Comparing Performance of Top Models'):
    relevant_models = {k: v for k,
                       v in model_performance.items() if k[0] in top_models and k[1] == DOWN_SAMPLE_RATE and k[2] == top_features[k[0]]}
    model_names = [model[0].split(' #')[0] for model in relevant_models]
    false_positives = [model['false_positives']
                       for _, model in relevant_models.items()]
    misclassifications = [model['misclassifications']
                          for _, model in relevant_models.items()]
    num_events = [model['num_events'] for _, model in relevant_models.items()]

    data = pd.concat([pd.DataFrame({
        'Model': model_names,
        'Metric': ['False Positive Rate' for _ in range(len(model_names))],
        'Value': [false_positives[i] / num_events[i] for i in range(len(false_positives))]
    }),
        pd.DataFrame({
            'Model': model_names,
            'Metric': ['Misclassification Rate' for _ in range(len(model_names))],
            'Value': [misclassifications[i] / num_events[i] for i in range(len(misclassifications))]
        })])

    sns.catplot(data,
                kind='bar',
                x='Model',
                y='Value',
                hue='Metric',
                palette=sns.color_palette(['#D94552', '#124F7B']))
    plt.title(title)
    plt.show()


def presentation_plot(model_performance,
                      title='Classification Time vs. Down Sample Rate'):
    _, ax = plt.subplots()

    relevant_models = {model: value for model, value in model_performance.items() if model[0]
                       in top_models and model[2] == 'none'}
    times = {x: list() for x in DOWN_SAMPLE_RATES}
    accuracies = {x: list() for x in DOWN_SAMPLE_RATES}

    for model, performance in relevant_models.items():
        times[model[1]].append(performance['time'])
        accuracies[model[1]].append(performance['accuracy'])

    mean_times = [np.mean(values) for _, values in times.items()]
    mean_accuracies = [np.mean(values) for _, values in accuracies.items()]
    accuracy_bins = list()
    for accuracy in mean_accuracies:
        if accuracy > 0.87:
            accuracy_bins.append('> 0.87')
        elif accuracy > 0.85:
            accuracy_bins.append('> 0.85')
        else:
            accuracy_bins.append('<= 0.85')

    df = pd.DataFrame({
        'Down Sample Rate': DOWN_SAMPLE_RATES,
        'Average Classification Time (s)': mean_times,
        'Average Accuracy': accuracy_bins
    })

    green = "#008001"
    yellow = "#FFA500"
    red = "#FF3128"

    # Create the color palette
    colors = [green, yellow, red]

    # Set the color palette in seaborn
    sns.set_palette(colors)

    sns.scatterplot(df,
                    x='Down Sample Rate',
                    y='Average Classification Time (s)',
                    hue='Average Accuracy',
                    label='Average Accuracy',
                    marker='o',
                    zorder=1)
    sns.lineplot(df,
                 x='Down Sample Rate',
                 y='Average Classification Time (s)',
                 color='#323A4C',
                 linewidth=1,
                 alpha=0.5,
                 zorder=0)

    # set x-axis label
    ax.set_xlabel('Down Sample Rate', fontsize=14)
    # set y-axis label
    ax.set_ylabel('Average Classification Time (s)',
                  fontsize=14)
    ax.set_title(title)
    ax.set_xscale('log')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    plt.show()
