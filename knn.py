import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def knn_basic_pipeline(num_neighbours):
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('classification', KNeighborsClassifier(num_neighbours))
    ])


def model_1():
    '''
    KNN #1: K = 4 (all features)
    '''
    return knn_basic_pipeline(4)


def model_2():
    '''
    KNN #2: K = 5 (all features)
    '''
    return knn_basic_pipeline(5)


def model_3():
    '''
    KNN #3: K = 6 (all features)
    '''
    return knn_basic_pipeline(6)


def model_4():
    '''
    KNN #4: K = 7 (all features)
    '''
    return knn_basic_pipeline(7)


def model_5():
    '''
    KNN #5: K = 8 (all features)
    '''
    return knn_basic_pipeline(8)


def model_6():
    '''
    KNN #6: K = 9 (all features)
    '''
    return knn_basic_pipeline(9)


def model_7():
    '''
    KNN #7: K = 10 (all features)
    '''
    return knn_basic_pipeline(10)


def model_8():
    '''
    KNN #8: K = 11 (all features)
    '''
    return knn_basic_pipeline(11)


def model_9():
    '''
    KNN #9: K = 12 (all features)
    '''
    return knn_basic_pipeline(12)
