"""
Bank Marketing Campaign - Term Deposit Prediction
MLOps Individual Assignment - Model Training Pipeline

Business Problem: Predict whether a bank client will subscribe to a term deposit
based on marketing campaign data, to improve targeting and reduce costs.
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
import mlflow
import mlflow.sklearn
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("STEP 1: DATA LOADING")
print("=" * 60)

df = pd.read_csv("data_raw/bank-additional.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\n--- Data Types ---")
print(df.dtypes)

print(f"\n--- Statistical Summary (Numeric) ---")
print(df.describe())

print(f"\n--- Target Variable Distribution ---")
print(df['y'].value_counts())
print(f"\nClass imbalance ratio: {df['y'].value_counts()['no'] / df['y'].value_counts()['yes']:.1f}:1")

print(f"\n--- Missing Values ---")
print(df.isnull().sum())

print(f"\n--- 'Unknown' Values in Categorical Columns ---")
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
for col in cat_cols:
    unknown_count = (df[col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"  {col}: {unknown_count} unknowns ({unknown_count/len(df)*100:.1f}%)")

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: DATA PREPROCESSING")
print("=" * 60)

# Encode target variable
le = LabelEncoder()
df['y_encoded'] = le.fit_transform(df['y'])  # no=0, yes=1
print(f"Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separate features and target
X = df.drop(columns=['y', 'y_encoded'])
y = df['y_encoded']

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Note about duration: keeping it as the assignment focuses on MLOps pipeline
# In a real scenario, duration would be removed as it's only known after the call
print("\nNote: 'duration' is kept for model building as the focus is on MLOps workflow.")
print("In production, this feature would cause data leakage and should be removed.")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")

# ============================================================
# 5. MODEL TRAINING & MLFLOW TRACKING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: MODEL TRAINING & MLFLOW TRACKING")
print("=" * 60)

# Set the experiment name
mlflow.set_experiment("bank-term-deposit-prediction")

with mlflow.start_run():
    # Build the full pipeline: preprocessing + model
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)
    print("Model training completed successfully!")

    # Log model parameters to MLflow
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("class_weight", "balanced")

    # ============================================================
    # 6. MODEL EVALUATION
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 60)

    # Predictions
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n--- Performance Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No (0)', 'Yes (1)']))

    print(f"\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted No  Predicted Yes")
    print(f"  Actual No      {cm[0][0]:>10}  {cm[0][1]:>13}")
    print(f"  Actual Yes     {cm[1][0]:>10}  {cm[1][1]:>13}")

    # Feature importance (top 15)
    feature_names = (numeric_features +
                     list(model_pipeline.named_steps['preprocessor']
                          .named_transformers_['cat']
                          .get_feature_names_out(categorical_features)))
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(15)
    print(f"\n--- Top 15 Feature Importances ---")
    for _, row in feat_imp.iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    # ============================================================
    # 7. SAVE MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: SAVING MODEL")
    print("=" * 60)

    # Save the full pipeline (preprocessor + model)
    joblib.dump(model_pipeline, 'model.pkl')
    print("Model saved as 'model.pkl'")

    # Save model metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'features': list(X.columns),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'target_classes': list(le.classes_),
        'metrics': {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4)
        },
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0])
    }

    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Model metadata saved as 'model_metadata.json'")

    # Save column info for the API
    columns_info = {
        'all_columns': list(X.columns),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'categorical_values': {}
    }
    for col in categorical_features:
        columns_info['categorical_values'][col] = list(df[col].unique())

    with open('columns_info.json', 'w') as f:
        json.dump(columns_info, f, indent=2)
    print("Column info saved as 'columns_info.json'")

    # Log model to MLflow
    mlflow.sklearn.log_model(model_pipeline, "model")
    print("Model logged to MLflow successfully!")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nFiles generated:")
print(f"  - model.pkl (trained model pipeline)")
print(f"  - model_metadata.json (model performance info)")
print(f"  - columns_info.json (feature info for API)")
