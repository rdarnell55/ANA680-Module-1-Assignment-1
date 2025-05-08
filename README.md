# ANA680 May2025 Module 1 - Assignment 2: Classification Models 
# Breast Cancer Classification Models

This repository contains the implementation and evaluation of eight supervised machine learning models on the Breast Cancer Wisconsin (Original) dataset. The work is part of **ANA680 - Machine Learning Deployment** at National University.

## Assignment Overview

**Objective**: Build and evaluate multiple classification models to predict whether a breast tumor is benign or malignant.

**Dataset**:  
- Source: [UCI ML Repository - Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))  
- Features: 10 numerical attributes describing cell characteristics  
- Target:  
  - `2` → Benign  
  - `4` → Malignant  
  (Recoded to `0` and `1` respectively)

## Models Implemented

- Logistic Regression  
- K-Nearest Neighbors (k = 5)  
- Support Vector Machine (Linear Kernel)  
- Support Vector Machine (RBF Kernel)  
- Naive Bayes  
- Decision Tree  
- Random Forest (n_estimators = 10)  
- XGBoost

All models are trained and tested using the same 75/25 split and evaluated using:
- **Accuracy**
- **Confusion Matrix**

## Results Summary

The models performed as follows:

| Model                      | Accuracy | Confusion Matrix         |
|---------------------------|----------|--------------------------|
| Logistic Regression       | 0.9600   | [[117, 1], [6, 51]]      |
| K-Nearest Neighbors (k=5) | 0.9771   | [[116, 2], [2, 55]]      |
| SVM (Linear Kernel)       | 0.9600   | [[116, 2], [5, 52]]      |
| SVM (RBF Kernel)          | 0.9657   | [[115, 3], [3, 54]]      |
| Naive Bayes               | 0.9600   | [[113, 5], [2, 55]]      |
| Decision Tree             | 0.9371   | [[115, 3], [8, 49]]      |
| Random Forest             | 0.9543   | [[115, 3], [5, 52]]      |
| XGBoost                   | 0.9600   | [[115, 3], [4, 53]]      |

## Notes
- Missing values were handled using SimpleImputer.
- Data was fetched directly using ucimlrepo to support reproducibility and deployment standards.

## Interpertation
- KNN and SVM (RBF Kernel) achieved the highest accuracy. However, differences in confusion matrices suggest trade-offs in false positives vs. false negatives, which are crucial in medical applications. Overall, this workflow supports model selection for future deployment.
