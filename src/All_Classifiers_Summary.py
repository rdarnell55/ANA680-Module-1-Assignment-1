#!/usr/bin/env python
# coding: utf-8
"""
All_Classifiers_Summary.py

This script performs a comparative evaluation of eight popular classification algorithms
on the Breast Cancer Wisconsin (Original) dataset using scikit-learn and XGBoost.

Workflow:
---------
1. Loads the dataset using the `ucimlrepo` package from the UCI Machine Learning Repository.
2. Handles missing values using mean imputation and splits the data into training and testing sets.
3. Defines a reusable evaluation function to measure accuracy and confusion matrix for any classifier.
4. Trains and tests the following models:
   - Logistic Regression
   - K-Nearest Neighbors (k=5)
   - SVM (Linear Kernel)
   - SVM (RBF Kernel)
   - Naive Bayes
   - Decision Tree
   - Random Forest (n_estimators=10)
   - XGBoost

5. Tabulates the performance results in a pandas DataFrame for comparison.

Output:
-------
- Prints the accuracy and confusion matrix for each model.
- (Optionally) Saves the results to a CSV for use in reports or Word documents.

This script is designed to support reproducibility and comparison of classifier performance in
a structured machine learning workflow.
"""

# In[23]:


# ============================
# 1. IMPORT LIBRARIES
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

# ============================
# 2. LOAD & PREP DATA
# ============================
dataset = fetch_ucirepo(id=15)
data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
data.rename(columns={'Class': 'Target'}, inplace=True)
data['Target'] = data['Target'].map({2: 0, 4: 1})

X = data.drop('Target', axis=1)
y = data['Target']

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = y[y.notna()]
X_imputed = X_imputed.loc[y.index]

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)

# ============================
# 3. DEFINE EVALUATION FUNCTION
# ============================
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return name, acc, cm

# ============================
# 4. TRAIN & EVALUATE ALL MODELS
# ============================
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (Linear Kernel)": SVC(kernel='linear'),
    "SVM (RBF Kernel)": SVC(kernel='rbf'),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []
for name, model in models.items():
    model_name, accuracy, cm = evaluate_model(model, name)
    results.append({
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Confusion Matrix': cm.tolist()
    })

# ============================
# 5. TABULATE RESULTS
# ============================
results_df = pd.DataFrame(results)
print("\n=== Classification Model Comparison Summary ===")
print(results_df)

# Optional: Save to CSV for Word
# results_df.to_csv("model_comparison_summary.csv", index=False)

