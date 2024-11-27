import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

class MLAlgorithms:
    def __init__(self, X, y=None, X_validation=None, y_validation=None, feature_names=None):
        self.X = X
        self.y = y
        self.X_validation = X_validation
        self.y_validation = y_validation

        self.feature_names = list(feature_names) if feature_names is not None else [f"Feature {i}" for i in range(X.shape[1])]

    def cross_validate(self, model, k_folds):
        print(f"Starting cross-validation for {model.__class__.__name__}...")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        best_model = None
        best_f1_score = -1

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

            scores['accuracy'].append(accuracy)
            scores['precision'].append(precision)
            scores['recall'].append(recall)
            scores['f1_score'].append(f1)

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = model

        avg_scores = {metric: np.mean(scores[metric]) for metric in scores}
        print(f"Completed cross-validation for {model.__class__.__name__}")
        return avg_scores, best_model

    def final_evaluation(self, model, model_name, selection_method):
        print(f"Evaluating {model_name} on final test data ({selection_method})...")
        y_pred = model.predict(self.X_validation)
        
        # Métricas generales
        accuracy = accuracy_score(self.y_validation, y_pred)
        precision = precision_score(self.y_validation, y_pred, average='weighted', zero_division=1)
        recall = recall_score(self.y_validation, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(self.y_validation, y_pred, average='weighted', zero_division=1)

        print("\nDetailed classification report:")
        report = classification_report(self.y_validation, y_pred, zero_division=1)
        print(report)

        # Save classification report
        with open(f"results/{model_name}_{selection_method}_classification_report.txt", "w") as f:
            f.write(report)
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(self.y_validation, y_pred)
        print(conf_matrix)

        # Save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=np.unique(self.y_validation), 
                    yticklabels=np.unique(self.y_validation))
        plt.title(f"Confusion Matrix for {model_name} ({selection_method})")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig(f"results/{model_name}_{selection_method}_confusion_matrix.png")
        plt.close()

        return report
