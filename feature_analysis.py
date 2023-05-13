import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def get_max_index(x):
    current_max = x[0]
    current_max_index = 0

    for i in range(len(x)):
        if x[i] > current_max:
            current_max = x[i]
            current_max_index = i

    return current_max_index


def get_min_index(x):
    current_min = x[0]
    current_min_index = 0

    for i in range(len(x)):
        if x[i] < current_min:
            current_min = x[i]
            current_min_index = i

    return current_min_index


def crossings(x):
    crossings = 0
    mean = np.mean(x)
    current = x[0]
    for i in range(len(x) - 1):
        next = x[i+1]
        crossings += current * next < mean
        current = next

    return crossings


def proportion_increasing(x):
    count_increasing = 0
    current = x[0]
    for i in range(len(x) - 1):
        next = x[i+1]
        count_increasing += current < next
        current = next

    return float(count_increasing) / len(x)


feature_list = [
    'min', 'max', 'mean', 'sd', 'median',
    'max_median_distance', 'min_median_distance',
    'proportion_above_mean', 'min_index',
    'max_index', 'crossings', 'proportion_increasing'
]

feature_map = {
    'min': min,
    'max': max,
    'mean': np.mean,
    'sd': np.std,
    'median': np.median,
    'max_median_distance': lambda x: abs(max(x) - np.median(x)),
    'min_median_distance': lambda x: abs(min(x) - np.median(x)),
    'proportion_above_mean': lambda x: np.mean(x > np.mean(x)),
    'min_index': get_min_index,
    'max_index': get_max_index,
    'crossings': lambda x: np.sum((x[1:] - np.median(x)) * (x[:-1] - np.median(x)) <= 0),
    'proportion_increasing': proportion_increasing
}


def get_features_from_data(training_data: pd.DataFrame, training: bool):
    excluded_columns = ['Unnamed: 0', 'label']
    if not training:
        excluded_columns.extend(['file_num', 'index'])
    return [name for name in training_data.columns if name not in excluded_columns]


def get_selected_features(classifier: Pipeline):
    if 'feature_selection' not in classifier.named_steps:
        return classifier.feature_names_in_

    input_features = classifier.feature_names_in_
    return input_features[classifier.named_steps['feature_selection'].support_]


def calculate_features(window: list):
    features = dict()
    for name, feature_function in feature_map.items():
        features[name] = feature_function(window)

    return features
