from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

NUM_FOLDS = 5


def model_1():
    '''
    Random Forest #1 (all features)
    '''
    return Pipeline([
        ('classification', RandomForestClassifier())
    ])


def model_2():
    '''
    Random Forest #2 (selected features)
    '''
    return Pipeline([
        ('feature_selection', RFE(RandomForestClassifier())),
        ('classification', RandomForestClassifier())
    ])
