# Music Genre Analysis

This repository contains an analysis of musical genres using supervised and unsupervised learning techniques. This project was developed as part of a **Machine Learning course** in my **Master's in Data Science** program. Both classification and clustering are addressed, employing the Kaggle Music Genre Classification Dataset.


## Overview

The analysis is divided into two main applications:

1. **Supervised Classification**: Application of probabilistic and non-probabilistic algorithms to predict musical genres.
2. **Clustering**: Use of unsupervised methods to group songs based on their musical characteristics, identifying patterns without requiring labels.

Supervised Classification utilize feature selection techniques.

---

## Dataset

- **Source**: [Kaggle Music Genre Classification Datasett](https://www.kaggle.com/datasets/purumalgi/music-genre-classification)

- **Instances**: 15,915 songs in the training set.
- **Features**: 17 musical attributes (e.g., 'popularity', 'danceability', 'energy', 'tempo').
- **Objective**:
  - **Classification**: Predict the musical genre (11 genres).
  - **Clustering**: Identify natural groupings based on musical attributes.

---

## Methodology

### Supervised Classification

#### Preprocessing
- Removal of duplicates.
- Analysis and handling of relevant outliers.
- Imputation of missing values.
- Standardization of numerical features.
- Class balancing using SMOTE and undersampling.
- Splitting the dataset into 80% training and 20% testing.

#### Algorithms Tested
- **Non-Probabilistic**:
  - k-Nearest Neighbors (k-NN)
  - Artificial Neural Networks (ANN)
  - Support Vector Machines (SVM)
  - Classification Trees
  - Rule Induction (RIPPER)
- **Probabilistic**:
  - Logistic Regression (Softmax)
  - Continuous Naive Bayes (Kernel Density Estimation)
  - Metaclassifiers (Gradient Boosting)

#### Feature Selection
Feature selection techniques were applied using the **WEKA program**, leveraging its tools for univariate, multivariate, and wrapper-based feature selection methods:
1. **InfoGain**: Univariate selection.
2. **CFS**: Multivariate selection.
3. **GreedyStepwise with Naive Bayes**: Wrapper method.

Cross-validation was used during training to evaluate model performance and fine-tune hyperparameters, followed by a final test set evaluation to assess the generalization ability of the models.

---

### Clustering

#### Preprocessing
- Application of the same preprocessing techniques as in classification, except for class balancing and dataset splitting.

#### Clustering Algorithms

- **Partitional Clustering**:
- **Hierarchical Clustering**:
- **Probabilistic Clustering**:

> **Note**: This section is still in progress and will be updated as the analysis advances.
