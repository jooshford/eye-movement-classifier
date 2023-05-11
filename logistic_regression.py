from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def logistic_regression_custom_pipeline(penalty='l1', C=1.0, solver='lbfgs', max_iter=10000):
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('classification', LogisticRegression(
            penalty=penalty, C=C, solver=solver, max_iter=max_iter))
    ])


def logistic_regression_elasticnet_custom_pipeline(C=1.0, l1_ratio=0.5, fit_intercept=True, solver='saga', max_iter=10000):
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('classification', LogisticRegression(penalty='elasticnet', C=C,
         l1_ratio=l1_ratio, fit_intercept=fit_intercept, solver=solver, max_iter=max_iter))
    ])


def model_1():
    '''
    Logistic Regression #3: L1 C=1.0 SAGA
    '''
    return logistic_regression_custom_pipeline(penalty='l1', C=1.0, solver='saga')


def model_2():
    """
    Logistic Regression #2: L2 C=1 LBFGS
    """
    return logistic_regression_custom_pipeline(penalty='l2', C=1, solver='lbfgs')


def model_3():
    """
    Logistic Regression #10: L2 C=1 Newton-CG
    """
    return logistic_regression_custom_pipeline(penalty='l2', C=1, solver='newton-cg')


def model_4():
    """
    Logistic Regression L2 #11: C=1, using sag solver
    """
    return logistic_regression_custom_pipeline(penalty='l2', C=1, solver='sag')


def model_5():
    """
    Logistic Regression L2 #12: C=1, using saga solver
    """
    return logistic_regression_custom_pipeline(penalty='l2', C=1, solver='saga')


def model_6():
    """
    Logistic Regression #6: Elasticnet ratio=0.25
    """
    return logistic_regression_elasticnet_custom_pipeline(C=1, l1_ratio=0.25)


def model_7():
    """
    Logistic Regression #7: Elasticnet ratio=0.5
    """
    return logistic_regression_elasticnet_custom_pipeline(C=1, l1_ratio=0.5)


def model_8():
    """
    Logistic Regression #8: Elasticnet ratio=0.75
    """
    return logistic_regression_elasticnet_custom_pipeline(C=1, l1_ratio=0.75)
