# [Model Tuning](https://github.com/CategorIAN/Naive-Bayes/blob/main/CrossValidation.py)

## Overview

This page explains how the Naive Bayes model is tuned in this project. The goal of tuning is to select hyperparameters 
that minimize prediction error on unseen data.

The two hyperparameters are:

- `n`: bin size used to discretize numeric features  
- `alpha`: Laplace smoothing strength  

---

## Grid Search

Grid search evaluates all combinations of hyperparameters:

- n ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}  
- α ∈ {0.01, 0.1, 0.5, 1, 2, 5, 10}  

Each combination is evaluated independently.

---

## 10-Fold Cross Validation

The dataset is partitioned into 10 approximately equal-sized subsets.

Each subset defines a *fold*. For each fold:

- 9 subsets are used for training  
- 1 subset is used for testing  

This process is repeated 10 times so that each subset is used as the test set exactly once.

For each sample in the test set, its predicted class is computed using the model trained on the corresponding training set.

---

## Error Computation

For each hyperparameter combination, the average error across all folds is computed.

Error is defined as:

error = 1 if prediction ≠ actual, else 0

This is equivalent to the **misclassification rate**.

---

## Summary

For each combination of hyperparameters (`n`, `alpha`) in the grid search:

For each fold `k`:

1. The model is trained on the training set for fold `k`  
2. The model makes predictions for each sample in the test set for fold `k`  
3. The error is computed using the misclassification rate  

After evaluating all folds, the error is averaged across the 10 folds to obtain the error for the hyperparameter combination (`n`, `alpha`).

The best hyperparameters are then selected as the pair that minimizes this average error.

---

## Model Selection

The best hyperparameters are those that minimize the average error across all folds.

---

## Where To See Results

- `Tuning The Model → Predict`: row-level predictions  
- `Tuning The Model → Error`: average error by `(n, alpha)`  
- `Tuning The Model → Best Model`: best hyperparameters found