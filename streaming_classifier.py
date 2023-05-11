from feature_analysis import (get_selected_features,
                              feature_map,
                              get_features_from_data)
import numpy as np
import pandas as pd


def classify(trained_classifier, window, previous_L, previous_R):
    feature_values = {
        'min': [0],
        'max': [0],
        'mean': [0],
        'sd': [0],
        'median': [0],
        'max_median_distance': [0],
        'min_median_distance': [0],
        'proportion_above_mean': [0],
        'min_index': [0],
        'max_index': [0],
        'crossings': [0],
        'proportion_increasing': [0],
        'previous_L': previous_L,
        'previous_R': previous_R
    }

    for name in get_selected_features(trained_classifier):
        if 'previous' not in name:
            feature_values[name][0] = feature_map[name](window)

    return trained_classifier.predict(pd.DataFrame(feature_values))


def train(classifier_pipeline, training_data: pd.DataFrame):
    X = training_data[get_features_from_data(training_data)]
    y = training_data['label']

    return classifier_pipeline.fit(X, y)
