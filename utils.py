import pandas as pd
from ProbabilisticAlg import Probabilistics
from NonProbabilisticAlg import NonProbabilistics

# Helper function for feature selection
def feature_selection(df, df_test, features):
    print(f"Selecting features: {features}")
    selected_columns = [df.columns[i] for i in features]
    X_selected = df[selected_columns]
    y = df['Class']
    X_t = df_test[selected_columns]
    y_t = df_test['Class']
    return X_selected, y, X_t, y_t

# Function to run and evaluate models
def run_all_models(X, y, X_validation, y_validation, selection_method):
    results1 = run_non_probabilistic_models(X, y, X_validation, y_validation, selection_method)
    results2 = run_probabilistic_models(X, y, X_validation, y_validation, selection_method)
    return results1, results2


def run_probabilistic_models(X, y, X_validation, y_validation, selection_method):
    print(f"\nStarting model evaluation for feature selection method: {selection_method}")
    pb_algorithms = Probabilistics(X, y, X_validation, y_validation)
    
    # Run models
    logistic_regression_model = pb_algorithms.logistic_regression(k_folds=5)
    gradient_boosting_model = pb_algorithms.gradient_boosting(k_folds=5)
    naive_bayes_kde_model = pb_algorithms.naive_bayes_kde(bandwidth=1.0, kernel="exponential", k_folds=5)
    
    # Create a list to store results
    results = []

    # Evaluate models and add results to the list
    lr_report = pb_algorithms.final_evaluation(logistic_regression_model, "Logistic Regression", selection_method)
    results.append({"Model": "Logistic Regression", "Selection Method": selection_method, "Report": lr_report})
    
    gb_report = pb_algorithms.final_evaluation(gradient_boosting_model, "Gradient Boosting", selection_method)
    results.append({"Model": "Gradient Boosting", "Selection Method": selection_method, "Report": gb_report})
    
    kde_nb_report = pb_algorithms.final_evaluation(naive_bayes_kde_model, "Naive Bayes KDE", selection_method)
    results.append({"Model": "Naive Bayes KDE", "Selection Method": selection_method, "Report": kde_nb_report})
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    # Display the table
    print("\nResults Summary:")
    print(results_df)

    # Save the results
    results_df.to_csv(f"results/{selection_method}_results.csv", index=False)
    
    print(f"Completed model evaluation for feature selection method: {selection_method}")
    return results_df


def run_non_probabilistic_models(X, y, X_validation, y_validation, selection_method):
    print(f"\nStarting model evaluation for feature selection method: {selection_method}")
    np_algorithms = NonProbabilistics(X, y, X_validation, y_validation)
    
    # Run models
    knn_model = np_algorithms.knn(k=10, k_folds=5)
    decision_tree_model = np_algorithms.decision_tree(k_folds=5)
    svm_model = np_algorithms.svm(k_folds=5)
    ann_model = np_algorithms.ann(k_folds=5)
    
    # Create a list to store results
    results = []

    # Evaluate models and add results to the list
    knn_report = np_algorithms.final_evaluation(knn_model, "k-NN", selection_method)
    results.append({"Model": "k-NN", "Selection Method": selection_method, "Report": knn_report})
    
    decision_tree_report = np_algorithms.final_evaluation(decision_tree_model, "Decision Tree", selection_method)
    results.append({"Model": "Decision Tree", "Selection Method": selection_method, "Report": decision_tree_report})
    
    svm_report = np_algorithms.final_evaluation(svm_model, "SVM", selection_method)
    results.append({"Model": "SVM", "Selection Method": selection_method, "Report": svm_report})
    
    ann_report = np_algorithms.final_evaluation(ann_model, "ANN", selection_method)
    results.append({"Model": "ANN", "Selection Method": selection_method, "Report": ann_report})
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    # Display the table
    print("\nResults Summary:")
    print(results_df)

    # Save the results
    results_df.to_csv(f"results/{selection_method}_results.csv", index=False)
    
    print(f"Completed model evaluation for feature selection method: {selection_method}")
    return results_df