import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class KDE_NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel
    
    def fit(self, X, y):
        # Identificar las clases y sus probabilidades a priori
        self.classes_ = np.unique(y)
        self.class_priors_ = {c: np.mean(y == c) for c in self.classes_}
        self.kde_models_ = {}
        
        # Entrenar un modelo KDE para cada característica en cada clase
        for c in self.classes_:
            X_c = X[y == c]  # Datos pertenecientes a la clase c
            kde_c = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(X_c.iloc[:, i].values.reshape(-1, 1)) 
                     for i in range(X.shape[1])]
            self.kde_models_[c] = kde_c
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self, ["kde_models_", "class_priors_"])
        log_probs = []
        
        # Calcular la probabilidad logarítmica para cada clase
        for c in self.classes_:
            log_prob = np.log(self.class_priors_[c])
            for i, kde in enumerate(self.kde_models_[c]):
                log_prob += kde.score_samples(X.iloc[:, i].values.reshape(-1, 1))
            log_probs.append(log_prob)
        
        # Convertir las probabilidades logarítmicas en probabilidades normales
        log_probs = np.array(log_probs).T
        log_probs -= log_probs.max(axis=1, keepdims=True)  # Evitar desbordamiento numérico
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        # Predecir la clase con la probabilidad más alta
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]