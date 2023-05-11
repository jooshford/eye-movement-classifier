from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


def SVC_basic_pipeline(kernel):
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('classification', SVC(kernel=kernel))
    ])


def model_1():
    '''
    SVM #1: Linear (all features)
    '''
    return SVC_basic_pipeline('linear')


def model_2():
    '''
    SVM #2: Poly (all features)
    '''
    return SVC_basic_pipeline('poly')


def model_3():
    '''
    SVM #3: RBF (all features)
    '''
    return SVC_basic_pipeline('rbf')


def model_4():
    '''
    SVM #4: Sigmoid (all features)
    '''
    return SVC_basic_pipeline('sigmoid')


def model_5():
    '''
    SVM #5: RBF (RFE features)
    '''
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('feature_selection', RFE(SVC(kernel='rbf'))),
        ('classification', SVC(kernel='rbf'))
    ])
