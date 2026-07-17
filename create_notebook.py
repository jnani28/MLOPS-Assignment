"""Script to generate the model_training.ipynb notebook from structured cells."""
import json

def make_md_cell(source):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def make_code_cell(source):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

# ===== TITLE =====
cells.append(make_md_cell("""# Bank Marketing Campaign - Term Deposit Prediction
## MLOps Individual Assignment

**Business Problem:** A leading bank runs marketing campaigns to offer term deposits. The goal is to predict which clients are most likely to subscribe, thereby reducing costs and increasing conversion rates.

**Dataset:** Bank Marketing Dataset — 4,119 rows × 21 features  
**Target Variable:** `y` (yes/no — subscribed to term deposit)"""))

# ===== IMPORTS =====
cells.append(make_md_cell("## 1. Import Libraries"))
cells.append(make_code_cell("""import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

print("All libraries imported successfully!")"""))

# ===== DATA LOADING =====
cells.append(make_md_cell("## 2. Data Loading"))
cells.append(make_code_cell("""df = pd.read_csv("bank-additional.csv")
print(f"Dataset shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
df.head()"""))

# ===== EDA =====
cells.append(make_md_cell("""## 3. Exploratory Data Analysis (EDA)

### 3.1 Data Types & Summary Statistics"""))

cells.append(make_code_cell("""df.dtypes"""))

cells.append(make_code_cell("""df.describe()"""))

cells.append(make_md_cell("### 3.2 Target Variable Distribution"))
cells.append(make_code_cell("""print("Target Variable Distribution:")
print(df['y'].value_counts())
print(f"\\nClass Imbalance Ratio: {df['y'].value_counts()['no'] / df['y'].value_counts()['yes']:.1f}:1")
print("\\nThe target is highly imbalanced (89% No vs 11% Yes).")
print("We will use class_weight='balanced' in our model to handle this.")"""))

cells.append(make_md_cell("### 3.3 Missing & Unknown Values"))
cells.append(make_code_cell("""print("Missing values per column:")
print(df.isnull().sum())
print("\\nNo null values found. However, some categorical columns contain 'unknown' values:")
print()
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
for col in cat_cols:
    unknown_count = (df[col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"  {col}: {unknown_count} unknowns ({unknown_count/len(df)*100:.1f}%)")
print("\\n'unknown' values are treated as a separate category (handled by OneHotEncoder).")"""))

# ===== PREPROCESSING =====
cells.append(make_md_cell("""## 4. Data Preprocessing

### 4.1 Encode Target Variable"""))
cells.append(make_code_cell("""le = LabelEncoder()
df['y_encoded'] = le.fit_transform(df['y'])  # no=0, yes=1
print(f"Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")"""))

cells.append(make_md_cell("### 4.2 Feature-Target Separation"))
cells.append(make_code_cell("""X = df.drop(columns=['y', 'y_encoded'])
y = df['y_encoded']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"\\nNote: 'duration' is kept as the assignment focuses on the MLOps pipeline.")
print("In production, duration causes data leakage and should be removed.")"""))

cells.append(make_md_cell("""### 4.3 Preprocessing Pipeline

We use a `ColumnTransformer` to:
- **StandardScaler** for numeric features
- **OneHotEncoder** for categorical features (handles 'unknown' as a category)"""))

cells.append(make_code_cell("""preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)
print("Preprocessing pipeline created.")"""))

# ===== TRAIN-TEST SPLIT =====
cells.append(make_md_cell("## 5. Train-Test Split"))
cells.append(make_code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\\nTraining target distribution:\\n{y_train.value_counts()}")
print(f"\\nTest target distribution:\\n{y_test.value_counts()}")"""))

# ===== MODEL TRAINING =====
cells.append(make_md_cell("""## 6. Model Training — Random Forest Classifier

We use `class_weight='balanced'` to handle the 8:1 class imbalance ratio."""))
cells.append(make_code_cell("""model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

model_pipeline.fit(X_train, y_train)
print("Model training completed successfully!")"""))

# ===== EVALUATION =====
cells.append(make_md_cell("## 7. Model Evaluation"))
cells.append(make_code_cell("""y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")"""))

cells.append(make_md_cell("### 7.1 Classification Report"))
cells.append(make_code_cell("""print(classification_report(y_test, y_pred, target_names=['No (0)', 'Yes (1)']))"""))

cells.append(make_md_cell("### 7.2 Confusion Matrix"))
cells.append(make_code_cell("""cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                 Predicted No  Predicted Yes")
print(f"  Actual No      {cm[0][0]:>10}  {cm[0][1]:>13}")
print(f"  Actual Yes     {cm[1][0]:>10}  {cm[1][1]:>13}")"""))

cells.append(make_md_cell("### 7.3 Feature Importances (Top 15)"))
cells.append(make_code_cell("""feature_names = (numeric_features +
                 list(model_pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_features)))
importances = model_pipeline.named_steps['classifier'].feature_importances_
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(15)

print("Top 15 Feature Importances:")
print("-" * 45)
for _, row in feat_imp.iterrows():
    bar = "█" * int(row['importance'] * 100)
    print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")"""))

# ===== SAVE MODEL =====
cells.append(make_md_cell("""## 8. Save Model

Save the complete pipeline (preprocessor + model) so the API can accept raw feature values directly."""))
cells.append(make_code_cell("""joblib.dump(model_pipeline, 'model.pkl')
print("Model saved as 'model.pkl'")

# Save metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'features': list(X.columns),
    'metrics': {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4)
    }
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Model metadata saved as 'model_metadata.json'")
print("\\nAll files ready for deployment!")"""))

# ===== CONCLUSION =====
cells.append(make_md_cell("""## Summary

| Step | Status |
|------|--------|
| Data Loading | ✅ Complete |
| EDA | ✅ Complete |
| Preprocessing | ✅ Complete |
| Model Training | ✅ Complete |
| Evaluation | ✅ Complete |
| Model Saving | ✅ Complete |

**Next Steps:** Deploy using Docker → AWS EC2 → Test inference via the API endpoint."""))

# ===== BUILD NOTEBOOK =====
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells
}

# Strip trailing newline from last line in each cell
for cell in notebook["cells"]:
    if cell["source"] and cell["source"][-1].endswith("\n"):
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

output_path = r"c:\Users\DLP-I516-75\OneDrive - Indian School of Business\Term 2\CT_bharani Sir\Assignment\model_training.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook created at: {output_path}")
