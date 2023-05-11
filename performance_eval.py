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
        self.predicted_B = list()

        self.true_N = list()
        self.true_L = list()
        self.true_R = list()
        self.true_B = list()
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
            elif true[i] == 3:
                self.predicted_B.append(predicted[i])
                self.true_B.append(true[i])

    def accuracy(self, type=None):
        if type == None:
            return accuracy_score(self.predicted, self.true)
        if type == 'N':
            return accuracy_score(self.predicted_N, self.true_N)
        if type == 'L':
            return accuracy_score(self.predicted_L, self.true_L)
        if type == 'R':
            return accuracy_score(self.predicted_R, self.true_R)
        if type == 'B':
            return accuracy_score(self.predicted_B, self.true_B)

    def confusion_matrix(self):
        return pd.DataFrame(confusion_matrix(self.true, self.predicted),
                            columns=['Predicted N',
                                     'Predicted L',
                                     'Predicted R',
                                     'Predicted B'],
                            index=['True N', 'True L', 'True R', 'True B']
                            )

    def __str__(self):
        output_string = f'overall accuracy: {self.accuracy():.3f}\n'
        output_string += f'non-event accuracy: {self.accuracy("N"):.3f}\n'
        output_string += f'left-look accuracy: {self.accuracy("L"):.3f}\n'
        output_string += f'right-look accuracy: {self.accuracy("R"):.3f}\n'
        output_string += f'blink accuracy: {self.accuracy("B"):.3f}\n\n'
        output_string += f'{self.confusion_matrix()}'

        return output_string


def run_n_times(model, training_data: pd.DataFrame, n=50):
    repeated_performance = list()
    for i in range(n):
        repeated_performance.append(cross_validate(training_data, model))

    return repeated_performance


def cross_validate(training_data: pd.DataFrame, model_function):
    X = training_data[get_features_from_data(training_data, training=True)]
    y = training_data['label']

    max_file = X['file_num'].max()

    predicted_list = list()
    true = list()

    k_folds = KFold(CV_NUM_FOLDS, shuffle=True)
    for train_files, test_files in k_folds.split(list(range(max_file+1))):
        train_index = np.where(X['file_num'].isin(train_files))
        test_index = np.where(X['file_num'].isin(test_files))
        X_train = X.iloc[train_index][get_features_from_data(
            training_data, training=False)]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index][get_features_from_data(
            training_data, training=False)]
        y_test = y.iloc[test_index]

        trained_classifier = model_function().fit(X_train, y_train)

        previous_L = 0
        previous_R = 0
        previous_B = 0
        for _, window in X_test.iterrows():
            window['previous_L'] = previous_L
            window['previous_R'] = previous_R
            window['previous_B'] = previous_B
            predicted = trained_classifier.predict(pd.DataFrame(window).T)
            previous_L = int(predicted == 1)
            previous_R = int(predicted == 2)
            previous_B = int(predicted == 3)
            predicted_list.extend(predicted)

        true.extend(y_test)

    print(ClassifierPerformance(predicted_list, true).confusion_matrix())

    return ClassifierPerformance(predicted_list, true)


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
