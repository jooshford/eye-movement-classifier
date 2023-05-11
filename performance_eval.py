import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from feature_analysis import get_features_from_data, get_selected_features
import streaming_classifier
from constants import *


class ClassifierPerformance:
    def __init__(self, predicted, true):
        self.predicted = predicted
        self.true = true

        self.predicted_N = list()
        self.predicted_L = list()
        self.predicted_R = list()

        self.true_N = list()
        self.true_L = list()
        self.true_R = list()
        for i in range(len(true)):
            if true[i] == 0:
                self.predicted_N.append(predicted[i])
                self.true_N.append(true[i])
            elif true[i] == 1:
                self.predicted_L.append(predicted[i])
                self.true_L.append(true[i])
            elif true[i] == 2:
                self.predicted_R.append(predicted[i])
                self.true_R.append(true[i])

    def accuracy(self, type=None):
        if type == None:
            return accuracy_score(self.predicted, self.true)
        if type == 'N':
            return accuracy_score(self.predicted_N, self.true_N)
        if type == 'L':
            return accuracy_score(self.predicted_L, self.true_L)
        if type == 'R':
            return accuracy_score(self.predicted_R, self.true_R)

    def confusion_matrix(self):
        return pd.DataFrame(confusion_matrix(self.true, self.predicted),
                            columns=['Predicted N',
                                     'Predicted L', 'Predicted R'],
                            index=['True N', 'True L', 'True R']
                            )

    def __str__(self):
        output_string = f'overall accuracy: {self.accuracy():.3f}\n'
        output_string += f'non-event accuracy: {self.accuracy("N"):.3f}\n'
        output_string += f'left-look accuracy: {self.accuracy("L"):.3f}\n'
        output_string += f'right-look accuracy: {self.accuracy("R"):.3f}\n\n'
        output_string += f'{self.confusion_matrix()}'

        return output_string


def run_n_times(model, training_data: pd.DataFrame, n=50):
    repeated_performance = list()
    for i in range(n):
        repeated_performance.append(cross_validate(training_data, model))

    return repeated_performance


def cross_validate(training_data: pd.DataFrame, model_function):
    X = training_data[get_features_from_data(training_data)]
    y = training_data['label']

    predicted = list()
    true = list()

    k_folds = KFold(CV_NUM_FOLDS, shuffle=True)
    for train_index, test_index in k_folds.split(X):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        trained_classifier = model_function().fit(X_train, y_train)

        predicted.extend(trained_classifier.predict(X_test))
        true.extend(y_test)

    print(ClassifierPerformance(predicted, true).confusion_matrix())

    return ClassifierPerformance(predicted, true)


def test_time(training_data: pd.DataFrame,
              classifier_pipeline: Pipeline,
              down_sample_rate,
              num_repeats=50):

    times = list()

    trained_classifier = streaming_classifier.train(classifier_pipeline,
                                                    training_data)

    current_previous = 0

    for _ in range(num_repeats):
        window = 500 + \
            np.random.randn(round(PROCESSED_SAMPLE_RATE *
                            WINDOW_TIME / down_sample_rate) * 20)
        start_time = time.time()
        current_previous = streaming_classifier.classify(
            trained_classifier,
            window,
            int(current_previous == 1),
            int(current_previous == 2))
        end_time = time.time()

        times.append(end_time - start_time)

    return np.mean(times)
