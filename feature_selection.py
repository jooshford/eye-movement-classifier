from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import LogisticRegression
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


def write_feature_selection(file_name, selection_object, training_data):
    X = training_data[get_features_from_data(training_data)]
    y = training_data['label']
    selection_object.fit(X, y)

    new_data = X.loc[:, selection_object[-1].support_]
    new_data['label'] = y
    new_data.to_csv(f'{TRAINING_DIRECTORY}/{file_name}.csv')


if __name__ == '__main__':
    training_data = pd.read_csv(f'{TRAINING_DIRECTORY}/1.csv')
    write_feature_selection('RFE', rfe_selector_1(), training_data)
