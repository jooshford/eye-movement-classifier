import knn
import svm
import mlp
import random_forest
import logistic_regression
import naive_bayes

models = {
    'Random Forest #1:': random_forest.model_1,
    'SVM #1: Linear': svm.model_1,
    'SVM #2: Poly': svm.model_2,
    'SVM #3: RBF': svm.model_3,
    'SVM #4: Sigmoid': svm.model_4,
    'KNN #1: K = 4': knn.model_1,
    'KNN #2: K = 5': knn.model_2,
    'KNN #3: K = 6': knn.model_3,
    'KNN #4: K = 7': knn.model_4,
    'KNN #5: K = 8': knn.model_5,
    'KNN #6: K = 9': knn.model_6,
    'KNN #7: K = 10': knn.model_7,
    'KNN #8: K = 11': knn.model_8,
    'KNN #9: K = 12': knn.model_9,
    'Logistic Regression #1: L1 SAGA': logistic_regression.model_1,
    'Logistic Regression #2: L2 LBFGS': logistic_regression.model_2,
    'Logistic Regression #3: L2 Newton-GC': logistic_regression.model_3,
    'Logistic Regression #4: L2 SAG': logistic_regression.model_4,
    'Logistic Regression #5: L2 SAGA': logistic_regression.model_5,
    'Logistic Regression #6: Elasticnet ratio=0.25': logistic_regression.model_6,
    'Logistic Regression #7: Elasticnet ratio=0.5': logistic_regression.model_7,
    'Logistic Regression #8: Elasticnet ratio=0.75': logistic_regression.model_8,
    'MLP #1: [10]': mlp.model_1,
    'MLP #2: [10, 10]': mlp.model_2,
    'MLP #3: [10, 10, 10]': mlp.model_3,
    'MLP #4: [20, 10] (tanh)': mlp.model_4,
    'MLP #5: [15, 5] (identity)': mlp.model_5,
    'MLP #6: [10, 20, 30] (logistic)': mlp.model_6,
    'MLP #7: [20, 30, 40] (relu)': mlp.model_7,
    'MLP #8: [30, 20] (tanh)': mlp.model_8,
    'MLP #9: [50] (relu)': mlp.model_9,
    'Naive Bayes #1: Gaussian': naive_bayes.model_1,
    'Naive Bayes #2: Bernoulli': naive_bayes.model_2
}

top_models = {
    'Random Forest #1:': random_forest.model_1,
    'SVM #3: RBF': svm.model_3,
    'KNN #2: K = 5': knn.model_2,
    'Logistic Regression #6: Elasticnet ratio=0.25': logistic_regression.model_6,
    'MLP #4: [20, 10] (tanh)': mlp.model_4,
    'Naive Bayes #2: Bernoulli': naive_bayes.model_2
}

top_features = {
    'Random Forest #1:': 'RFE_LR',
    'SVM #3: RBF': 'none',
    'KNN #2: K = 5': 'RFE_LR',
    'Logistic Regression #6: Elasticnet ratio=0.25': 'none',
    'MLP #4: [20, 10] (tanh)': 'none',
    'Naive Bayes #2: Bernoulli': 'RFE_LR'
}
