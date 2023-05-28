from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif, SelectFpr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from feature_analysis import get_features_from_data
import pandas as pd
from constants import *


def rfe_selector_1():
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('feature_selection', RFE(LogisticRegression()))
    ])


def rfe_selector_2():
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('feature_selection', RFE(RandomForestClassifier()))
    ])


def rfe_selector_3():
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('feature_selection', RFE(LinearSVC()))
    ])


def select_k_best_1():
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif))
    ])


def select_k_best_2():
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('feature_selection', SelectKBest(mutual_info_classif))
    ])


def no_algorithm():
    return Pipeline([
        ('preprocessing', StandardScaler())
    ])


selection_methods = {
    'RFE_LR': rfe_selector_1(),
    'RFE_RF': rfe_selector_2(),
    'RFE_SVM': rfe_selector_3(),
    'f_classif': select_k_best_1(),
    'mutual_info': select_k_best_2(),
    'none': no_algorithm()
}


def write_feature_selection(file_name, selection_object, training_data):
    X = training_data[get_features_from_data(training_data)]
    file_nums = training_data['file_num']
    indexes = training_data['index']
    y = training_data['label']
    selection_object.fit(X, y)

    new_data = X.loc[:, selection_object[-1].get_support()]
    new_data['label'] = y
    new_data['file_num'] = file_nums
    new_data['index'] = indexes

    for attribute in ['previous_L', 'previous_R', 'previous_B']:
        if attribute not in new_data.columns:
            new_data[attribute] = training_data[attribute]
    new_data.to_csv(f'{TRAINING_DIRECTORY}/{file_name}.csv', index=False)


if __name__ == '__main__':
    training_data = pd.read_csv(f'{TRAINING_DIRECTORY}/{DOWN_SAMPLE_RATE}.csv')
    write_feature_selection('RFE_LR', rfe_selector_1(), training_data)
    write_feature_selection('RFE_RF', rfe_selector_2(), training_data)
    write_feature_selection('RFE_SVM', rfe_selector_3(), training_data)
    write_feature_selection('f_classif', select_k_best_1(), training_data)
    write_feature_selection('RFE_LR', rfe_selector_1(), training_data)
    write_feature_selection('none', no_algorithm(), training_data)
