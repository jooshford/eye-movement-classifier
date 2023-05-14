from feature_analysis import (get_selected_features,
                              feature_map,
                              get_features_from_data)
import numpy as np
import pandas as pd


def classify(trained_classifier, window, previous_L, previous_R, previous_B,
             selected_features):

    feature_values = {name: [feature_map[name](
        window)] for name in selected_features if 'previous' not in name}
    feature_values['previous_L'] = previous_L
    feature_values['previous_R'] = previous_R
    feature_values['previous_B'] = previous_B

    print(feature_values)
    input()

    return trained_classifier.predict(pd.DataFrame(feature_values))


def train(classifier_pipeline, training_data: pd.DataFrame):
    X = training_data[get_features_from_data(training_data, False)]
    y = training_data['label']

    return classifier_pipeline.fit(X, y)
