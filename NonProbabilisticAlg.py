import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from MLAlgorithms import MLAlgorithms

class NonProbabilistics(MLAlgorithms):
    def __init__(self, X, y, X_validation, y_validation):

        super().__init__(X, y, X_validation, y_validation)

    def knn(self, k_folds=5, k=10, weights='distance', metric='minkowski'):
        print("Running k-NN model...")
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
        results, model = self.cross_validate(knn, k_folds)
        print("k-NN model completed.")
        return model

    def decision_tree(self, k_folds=5, max_depth=20, min_samples_split=5, min_samples_leaf=4, criterion='gini'):
        print("Running Decision Tree model...")
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)
        results, model = self.cross_validate(tree, k_folds)
        print("Decision Tree model completed.")
        return model

    def svm(self, k_folds=5, kernel='rbf', C=10.0, gamma='auto'):
        print("Running SVM model...")
        svm = SVC(kernel=kernel, C=C, gamma=gamma)
        results, model = self.cross_validate(svm, k_folds)
        print("SVM model completed.")
        return model

    def ann(self, k_folds=5, hidden_layer_sizes=(300, 200, 50), activation='tanh', solver='adam', max_iter=1000):
        print("Running ANN model...")
        ann = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter)
        results, model = self.cross_validate(ann, k_folds)
        print("ANN model completed.")
        return model