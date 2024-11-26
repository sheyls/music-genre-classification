import pandas as pd
import numpy as np
import wittgenstein as lw
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
print("Loading datasets...")
df_train = pd.read_csv('dataset/final_train_data.csv')
df_test = pd.read_csv('dataset/final_test_data.csv')
print("Datasets loaded.")

# Filter to keep only Rock (10) and Pop (9) classes
print("Filtering dataset for Rock and Pop classes...")
df_train_filtered = df_train[df_train['Class'].isin([10, 9])].copy()
df_test_filtered = df_test[df_test['Class'].isin([10, 9])].copy()

# Convert classes to binary (1 for Rock, 0 for Pop)
df_train_filtered['Class'] = df_train_filtered['Class'].apply(lambda x: 1 if x == 10 else 0)
df_test_filtered['Class'] = df_test_filtered['Class'].apply(lambda x: 1 if x == 10 else 0)


X_train = df_train_filtered.drop(columns=['Class'])
y_train = df_train_filtered['Class']
X_test = df_test_filtered.drop(columns=['Class'])
y_test = df_test_filtered['Class']
print("Dataset transformed to binary classes for Rule Induction.")


# Initialize results DataFrame
results_df = pd.DataFrame(columns=["Model", "Feature Selection", "Accuracy", "Precision", "Recall", "F1 Score"])

# Function to evaluate Rule Induction model with RIPPER
def evaluate_rule_induction(X_train, y_train, X_test, y_test, selection_method):
    print(f"\nStarting RIPPER model evaluation for {selection_method}...")
    # Convert the dataset back to include the target in the same DataFrame as required by wittgenstein
    df_train_rule = X_train.copy()
    df_train_rule['Class'] = y_train

    # Initialize RIPPER model
    ripper = lw.RIPPER()
    ripper.fit(df_train_rule, class_feat='Class')


    # Print the extracted rules
    print("\nExtracted Rules:")
    for rule in ripper.ruleset_.rules:
        print(rule)

    # Predict on test set
    y_pred = ripper.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    # Append results to DataFrame
    results_df.loc[len(results_df)] = ["RIPPER Rule Induction", selection_method, accuracy, precision, recall, f1]
    print(f"RIPPER model evaluated on {selection_method}.")

# Evaluate RIPPER with all original features
evaluate_rule_induction(X_train, y_train, X_test, y_test, "All Features")

# Helper function for feature selection
def feature_selection(df, df_test, features):
    print(f"Selecting features: {features}")
    selected_columns = [df.columns[i] for i in features]
    X_selected = df[selected_columns]
    y = df['Class']
    X_t = df_test[selected_columns]
    y_t = df_test['Class']
    return X_selected, y, X_t, y_t

# Feature Selections and Evaluations
feature_selections = {
    "Univariate Selection": np.sort(np.array([2,7,11,3,1,4,10]) - 1).tolist(),
    "Multivariate Selection": np.sort(np.array([1,2,3,5,6,7,11]) - 1).tolist(),
    "Wrapper Selection": np.sort(np.array([2,7,9,10,1]) - 1).tolist()
}

# Run RIPPER for each feature selection
for selection_method, features in feature_selections.items():
    X_train_sel, y_train_sel, X_test_sel, y_test_sel = feature_selection(df_train_filtered, df_test_filtered, features)
    evaluate_rule_induction(X_train_sel, y_train_sel, X_test_sel, y_test_sel, selection_method)

# Display results
print("\nFinal Results:")
print(results_df.to_string(index=False))

# Plotting the results
print("Plotting results...")
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Feature Selection", y="F1 Score", hue="Model")
plt.title("RIPPER Model Performance for Rock vs. Pop")
plt.xlabel("Feature Selection Method")
plt.ylabel("F1 Score")
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.show()