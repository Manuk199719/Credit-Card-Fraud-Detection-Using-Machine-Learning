# Credit Card Fraud Detection

A machine learning project exploring credit card transaction fraud through data cleaning, exploratory data analysis, supervised classification, clustering, dimensionality reduction, and anomaly detection.

## Project at a glance

- **Dataset:** 284,807 transactions and 31 columns
- **Target:** `Class` — `0` = normal, `1` = fraudulent
- **Fraud prevalence:** 0.173% (492 fraud cases)
- **Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn
- **Main challenge:** Severe class imbalance

The dataset contains anonymized PCA-derived features (`V1`–`V28`), plus `Time`, `Amount`, and `Class`. 

## Objectives

- Inspect data types, missing values, duplicates, and descriptive statistics.
- Explore transaction amounts, class distribution, outliers, and feature correlations.
- Train and evaluate supervised fraud classifiers.
- Explore clustering, dimensionality reduction, and anomaly detection.

## Selected visualizations

### Class imbalance

![Transaction class distribution](images/class_distribution.png)

Only **0.173%** of transactions are labelled as fraud. Accuracy alone can be misleading because a model can perform well on the majority class while missing fraudulent transactions.

### Transaction amount distribution

![Transaction amount distribution](images/amount_distribution.png)

This chart displays amounts up to the 99th percentile to make the main distribution easier to inspect. Higher amounts are omitted from this visualization only, not from the dataset.

### Feature correlation

![Correlation heatmap](images/correlation_heatmap.png)

Pearson correlations for selected features. Correlation indicates linear association, not causation.

## Workflow

### 1. Data preparation and exploratory analysis
- Load the CSV using Pandas and inspect its shape, schema, summary statistics, missing values, and duplicate rows.
- Explore class distribution, transaction amounts, potential outliers, `Time` versus `Amount`, and feature correlations.
- For cleaning practice, missing values can be introduced deliberately and then imputed; preserve the original dataset for reproducibility.

### 2. Supervised learning
Models explored:
- Logistic Regression
- Decision Tree
- Random Forest

Evaluation:
- Confusion matrix
- Precision, recall, and F1-score
- ROC curve and ROC-AUC
- Class weighting to investigate changes in fraud recall

**Model selection:** Choose the model based on the actual test-set metrics and the trade-off between missed fraud and false alerts. Do not choose by accuracy alone.

### 3. Unsupervised learning
- **K-Means:** Explore transaction groupings and use the Elbow Method to guide the cluster count.
- **PCA:** Reduce scaled features to two principal components for visualization; use `Class` only to colour the plot after fitting PCA.
- **Isolation Forest:** Detect unusual transactions without using `Class` during training, then compare anomaly flags with actual labels for evaluation.

Anomalies are not necessarily fraud, and clusters do not automatically correspond to fraud categories.

## Getting started

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Load the dataset

Place `creditcard.csv` beside the notebook, then run:

```python
import pandas as pd

df = pd.read_csv("creditcard.csv")
print("Shape:", df.shape)
display(df.head())
df.info()
```

### Separate features and target

Exclude `Transaction_Type` if it was created from `Class`; otherwise, it leaks the answer into the model inputs.

```python
X = df.drop(columns=["Class", "Transaction_Type"], errors="ignore")
y = df["Class"]
```

### Create a stratified train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Fit scalers and other preprocessing steps on the training set only, then apply the fitted transformations to the test set.

## Key takeaways

- Fraud detection is a highly imbalanced classification problem.
- Precision, recall, F1-score, ROC-AUC, and optionally PR-AUC provide more insight than accuracy alone.
- Class weighting may improve fraud recall but can increase false positives.
- Clustering and anomaly detection provide exploratory signals, not definitive fraud labels.

## Repository structure

```text
.
├── README.md
├── credit_card_fraud_detection.ipynb
├── creditcard.csv              # Keep local; do not commit raw data
└── images/
    ├── class_distribution.png
    ├── amount_distribution.png
    └── correlation_heatmap.png
```

## Notes

- Exact model scores depend on the notebook's preprocessing, split, and model settings. Report the values produced by your run; no single algorithm is guaranteed to perform best.
- Do not commit the raw dataset to a public repository unless its licence and sharing terms permit it. Link to the authorised source instead.
