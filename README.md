# ANA680 May2025 Module 1 - Assignment 2: Building Classification Models Breast Cancer Classification Models

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

## 📁 Repository Structure


