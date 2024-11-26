from NB_KDE import KDE_NaiveBayes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier

import pandas as pd
import numpy as np

from MLAlgorithms import MLAlgorithms
    
class Probabilistics(MLAlgorithms):
    def __init__(self, X, y, X_validation, y_validation):
        super().__init__(X, y, X_validation, y_validation)
    
    def logistic_regression(self, k_folds=5):
        lr = LogisticRegression(
            solver='lbfgs',
            C=1.0,
            max_iter=1000,
            class_weight='balanced'
        )
        results, model = self.cross_validate(lr, k_folds)
        return model
    
    def naive_bayes_kde(self, bandwidth=1.0, kernel="exponential", k_folds=5):
        best_bandwidth = 0.5
        best_kernel = 'epanechnikov'

        kde_nb = KDE_NaiveBayes(
            bandwidth=best_bandwidth,
            kernel=best_kernel
        )

        # kde_nb = KDE_NaiveBayes(bandwidth=bandwidth, kernel=kernel)
        results, model = self.cross_validate(kde_nb, k_folds)
        return model 
    
    def gradient_boosting(self, k_folds=5):
        gb = XGBClassifier(eval_metric="mlogloss")
        N_CLASES = 11 

        gb = XGBClassifier(
            objective='multi:softprob',
            num_class=N_CLASES,
            eval_metric='mlogloss',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
           # scale_pos_weight=1
        )
        results, model = self.cross_validate(gb, k_folds)
        return model