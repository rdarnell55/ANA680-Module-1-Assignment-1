# Accessing the Breast Cancer Wisconsin Dataset

This guide provides instructions and example code for loading the **Breast Cancer Wisconsin (Original)** dataset used in the classification model evaluation project.

---

## Dataset Source

UCI Machine Learning Repository:  
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

---

## Recommended Access Method (for reproducibility)

Use the `ucimlrepo` Python package to fetch the dataset programmatically.

### Installation

```bash
pip install ucimlrepo
```

### Fetching the Dataset

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset by ID (15 corresponds to Breast Cancer Wisconsin Original)
dataset = fetch_ucirepo(id=15)

# Combine features and target labels into one DataFrame
data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

# Rename the target column and recode labels (2 = Benign, 4 = Malignant)
data.rename(columns={'Class': 'Target'}, inplace=True)
data['Target'] = data['Target'].map({2: 0, 4: 1})

# Now ready to use
print(data.head())
```

---

## Why use `ucimlrepo`?

- Automatically pulls from a trusted, versioned source.
- Simplifies dataset fetching and reduces manual errors.
- Excellent for reproducibility in deployment-focused coursework.
