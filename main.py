import pandas as pd
from utils import feature_selection, run_all_models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

def extract_accuracy(report):
    match = re.search(r"accuracy\s+([\d\.]+)", report)
    if match:
        return float(match.group(1))
    return None

def plot_results(df):
    df["Accuracy"] = df["Report"].apply(extract_accuracy)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Selection Method", y="Accuracy", hue="Model", ci=None)

    plt.title("Accuracy by Method Selection and Model", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Selection Method", fontsize=12)
    plt.legend(title="Model", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/final_accuracy_by_method.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    # Load datasets
    print("Loading datasets...")
    df_train = pd.read_csv('dataset/standard_final_train_data.csv')
    #df_train = pd.read_csv('dataset/final_test_data.csv')

    df_test = pd.read_csv('dataset/standard_final_test_data.csv')
    #df_test = pd.read_csv('dataset/final_train_data.csv')

    X_test_final = df_test.drop(columns=['Class'])
    y_test_final = df_test['Class']
    print("Datasets loaded.")

    # Using all original features
    print("Evaluating with all original features...")
    results_non, results_prop = run_all_models(df_train.drop(columns=['Class']), df_train['Class'], X_test_final, y_test_final, "All Features")

    # The feature selection was made using WEKA.
    # Univariate Feature Selection
    print("Evaluating with univariate feature selection...")
    uni_features = np.sort(np.array([11,7,3,4,1,6,2,9])-1).tolist()
    X, y, X_t, y_t = feature_selection(df_train, df_test, uni_features)
    results_non_uni, results_prop_uni = run_all_models(X, y, X_t, y_t, "Univariate Selection")

    # Multivariate Feature Selection
    print("Evaluating with multivariate feature selection...")
    multi_features = np.sort(np.array([11,7,3,1,6,2,4,9])-1).tolist()
    X, y, X_t, y_t = feature_selection(df_train, df_test, multi_features)
    results_non_multi, results_prop_multi = run_all_models(X, y, X_t, y_t, "Multivariate Selection")

    # Wrapper Feature Selection
    print("Evaluating with wrapper feature selection...")
    wrapper_features = np.sort(np.array([1,2,3,4,5,6,9,11])-1).tolist()
    X, y, X_t, y_t = feature_selection(df_train, df_test, wrapper_features)
    results_non_wrapper, results_prop_wrapper = run_all_models(X, y, X_t, y_t, "Wrapper Selection")

    # Display Results Table
    final_results = pd.concat([results_non, results_prop, results_non_uni, results_prop_uni, results_non_multi, results_prop_multi, results_non_wrapper, results_prop_wrapper], ignore_index=True)

    print("\nFinal Results Summary:")
    print(final_results)

    final_results.to_csv("results/final_results_summary.csv", index=False)

    plot_results(final_results)
