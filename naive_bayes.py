from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def model_1():
    '''
    Naive Bayes #1
    '''
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('classification', GaussianNB())
    ])


def model_2():
    '''
    Naive Bayes #2 (feature selection using VarianceThreshold)
    '''
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('classification', BernoulliNB(alpha=1.0))
    ])
