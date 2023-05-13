from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder


def mlp_custom_pipeline(hidden_layer_sizes, activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=10000):
    return Pipeline([
        ('preprocessing', StandardScaler()),
        ('classifier', MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
         solver=solver, alpha=alpha, learning_rate_init=learning_rate_init, max_iter=max_iter))
    ])


def model_1():
    '''
    MLP #1: 1 hidden layer with 10 neurons
    '''
    return mlp_custom_pipeline((10,))


def model_2():
    '''
    MLP #2: 2 hidden layers with 10 neurons each
    '''
    return mlp_custom_pipeline((10, 10))


def model_3():
    '''
    MLP #3: 3 hidden layers with 10 neurons each
    '''
    return mlp_custom_pipeline((10, 10, 10))


def model_4():
    '''
    MLP #4: 2 hidden layers with 20 and 10 neurons, using 'tanh' activation
    '''
    return mlp_custom_pipeline((20, 10), activation='tanh')


def model_5():
    '''
    MLP #5: 2 hidden layers with 15 and 5 neurons, using 'identity' activation and 'lbfgs' solver
    '''
    return mlp_custom_pipeline((15, 5), activation='identity', solver='lbfgs')


def model_6():
    '''
    MLP #6: 3 hidden layers with 10, 20, and 30 neurons, using 'logistic' activation and 'sgd' solver
    '''
    return mlp_custom_pipeline((10, 20, 30), activation='logistic', solver='sgd')


def model_7():
    '''
    MLP #7: 3 hidden layers with 20, 30, and 40 neurons, using 'relu' activation, 'sgd' solver,
           and a learning rate of 0.01
    '''
    return mlp_custom_pipeline((20, 30, 40), activation='relu', solver='sgd', learning_rate_init=0.01)


def model_8():
    '''
    MLP #8: 2 hidden layers with 30 and 20 neurons, using 'tanh' activation, 'adam' solver,
           and L2 regularization (alpha) of 0.001
    '''
    return mlp_custom_pipeline((30, 20), activation='tanh', solver='adam', alpha=0.001)


def model_9():
    '''
    MLP #9: 1 hidden layer with 50 neurons, using 'relu' activation, 'adam' solver,
           and a learning rate of 0.005
    '''
    return mlp_custom_pipeline((50,), activation='relu', solver='adam', learning_rate_init=0.005)
