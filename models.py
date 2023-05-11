import knn
import svm
import random_forest
import logistic_regression

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
    'Logistic regression #1: L1 SAGA': logistic_regression.model_1,
    'Logistic regression #2: L2 LBFGS': logistic_regression.model_2,
    'Logistic regression #3: L2 Newton-GC': logistic_regression.model_3,
    'Logistic regression #4: L2 SAG': logistic_regression.model_4,
    'Logistic regression #5: L2 SAGA': logistic_regression.model_5,
    'Logistic regression #6: Elasticnet ratio=0.25': logistic_regression.model_6,
    'Logistic regression #7: Elasticnet ratio=0.5': logistic_regression.model_7,
    'Logistic regression #8: Elasticnet ratio=0.75': logistic_regression.model_8,
}

top_models = {
    'Random Forest #1:': random_forest.model_1,
    'SVM #3: RBF': svm.model_3,
    'KNN #10: K = 10': knn.model_7,
    'Logistic regression #33: Elasticnet ratio=0.25': logistic_regression.model_6,
}
