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

    def count_false_positives(self):
        false_positives = 0
        previous_predicted = 0
        event_happened = False
        for i in range(len(self.predicted)):
            current_predicted = self.predicted[i]
            current_true = self.true[i]
            if current_predicted == 0:
                if event_happened == False and previous_predicted != 0:
                    false_positives += 1

                event_happened = False
                previous_predicted = current_predicted
                continue

            if current_true != 0:
                event_happened = True

            previous_predicted = current_predicted

        return false_positives

    def count_events(self):
        event_count = 0
        previous_label = 0
        for label in self.true:
            if label != 0 and previous_label == 0:
                event_count += 1

        return event_count

    def count_misclassified_events(self):
        misclassified_events = 0
        previous_true = 0
        classified_event = False
        for i in range(len(self.predicted)):
            current_predicted = self.predicted[i]
            current_true = self.true[i]
            if current_true == 0:
                if classified_event == False and previous_true != 0:
                    misclassified_events += 1

                classified_event = False
                previous_true = current_true
                continue

            if current_predicted == current_true:
                classified_event = True

            previous_true = current_true

        return misclassified_events

    def confusion_matrix(self):
        return pd.DataFrame(confusion_matrix(self.true, self.predicted),
                            columns=['Predicted N',
                                     'Predicted L',
                                     'Predicted R',
                                     'Predicted B'],
                            index=['True N', 'True L', 'True R', 'True B'])

    def __str__(self):
        output_string = f'overall accuracy: {self.accuracy():.3f}\n'
        output_string += f'non-event accuracy: {self.accuracy("N"):.3f}\n'
        output_string += f'left-look accuracy: {self.accuracy("L"):.3f}\n'
        output_string += f'right-look accuracy: {self.accuracy("R"):.3f}\n'
        output_string += f'blink accuracy: {self.accuracy("B"):.3f}\n\n'
        output_string += f'{self.confusion_matrix()}'

        return output_string


def run_n_times(model_functions, training_data: pd.DataFrame, n=50):
    repeated_performances = [list() for _ in range(len(model_functions))]
    for i in range(n):
        print(f'Iteration {i}')
        performances = cross_validate(training_data, model_functions)
        for i in range(len(model_functions)):
            repeated_performances[i].append(performances[i])

    return repeated_performances


def cross_validate(training_data: pd.DataFrame, model_functions):
    X = training_data[get_features_from_data(training_data, True)]
    y = training_data['label']

    max_file = X['file_num'].max()

    predicted_lists = [list() for _ in range(len(model_functions))]
    true = list()

    k_folds = KFold(CV_NUM_FOLDS, shuffle=True)
    for train_files, test_files in k_folds.split(list(range(max_file+1))):
        train_index = np.where(X['file_num'].isin(train_files))
        test_index = np.where(X['file_num'].isin(test_files))
        X_train = X.iloc[train_index][get_features_from_data(
            training_data, False)]
        y_train = y.iloc[train_index]
        test_data = training_data.iloc[test_index].sort_values(
            ['file_num', 'index'])
        X_test = test_data[get_features_from_data(training_data, False)]
        y_test = test_data['label']

        trained_classifiers = [model_function().fit(
            X_train, y_train) for model_function in model_functions]

        previous_L = [0 for _ in range(len(model_functions))]
        previous_R = [0 for _ in range(len(model_functions))]
        previous_B = [0 for _ in range(len(model_functions))]
        for _, row in X_test.iterrows():
            window = row.copy()
            window_dataframe = pd.DataFrame(window).T
            for i in range(len(model_functions)):
                window_dataframe['previous_L'] = [previous_L[i]]
                window_dataframe['previous_R'] = [previous_R[i]]
                window_dataframe['previous_B'] = [previous_B[i]]
                predicted = trained_classifiers[i].predict(window_dataframe)
                previous_L[i] = int(predicted == 1)
                previous_R[i] = int(predicted == 2)
                previous_B[i] = int(predicted == 3)
                predicted_lists[i].extend(predicted)

        true.extend(y_test)

    return [ClassifierPerformance(predicted_list, true) for predicted_list in predicted_lists]


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
