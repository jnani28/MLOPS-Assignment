"""
Bank Marketing Campaign - Term Deposit Prediction
MLOps Individual Assignment - Model Training Pipeline

Business Problem: Predict whether a bank client will subscribe to a term deposit
based on marketing campaign data, to improve targeting and reduce costs.
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data_raw") / "bank-additional.csv"
MODEL_PATH = Path("model.pkl")
METADATA_PATH = Path("model_metadata.json")
COLUMNS_INFO_PATH = Path("columns_info.json")
TARGET_COL = "y"
# Prevent target leakage: call duration is only known after outcome-affecting interaction.
LEAKAGE_COLUMNS = ["duration"]

# ============================================================
# 1. DATA LOADING
# ============================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("STEP 1: DATA LOADING")
    logger.info("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("First 5 rows:\n%s", df.head().to_string())

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    logger.info("\n--- Data Types ---\n%s", df.dtypes)
    logger.info("\n--- Statistical Summary (Numeric) ---\n%s", df.describe())

    logger.info("\n--- Target Variable Distribution ---\n%s", df[TARGET_COL].value_counts())
    imbalance = df[TARGET_COL].value_counts()["no"] / df[TARGET_COL].value_counts()["yes"]
    logger.info("Class imbalance ratio: %.1f:1", imbalance)

    logger.info("\n--- Missing Values ---\n%s", df.isnull().sum())

    logger.info("\n--- 'Unknown' Values in Categorical Columns ---")
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        unknown_count = (df[col] == "unknown").sum()
        if unknown_count > 0:
            logger.info("  %s: %s unknowns (%.1f%%)", col, unknown_count, unknown_count / len(df) * 100)

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 3: DATA PREPROCESSING")
    logger.info("=" * 60)

    # Encode target variable
    le = LabelEncoder()
    df["y_encoded"] = le.fit_transform(df[TARGET_COL])  # no=0, yes=1
    logger.info("Target encoding: %s", dict(zip(le.classes_, le.transform(le.classes_))))

    # Separate features and target and remove leakage-prone columns.
    X = df.drop(columns=[TARGET_COL, "y_encoded"] + LEAKAGE_COLUMNS, errors="ignore")
    y = df["y_encoded"]

    # Identify column types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    logger.info("Numeric features (%s): %s", len(numeric_features), numeric_features)
    logger.info("Categorical features (%s): %s", len(categorical_features), categorical_features)
    logger.info("Removed leakage-prone features: %s", LEAKAGE_COLUMNS)

# Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 4: TRAIN-TEST SPLIT")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logger.info("Training set: %s samples", X_train.shape[0])
    logger.info("Test set: %s samples", X_test.shape[0])
    logger.info("Training target distribution:\n%s", y_train.value_counts())
    logger.info("Test target distribution:\n%s", y_test.value_counts())

# ============================================================
# 5. MODEL TRAINING & MLFLOW TRACKING
# ============================================================
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 5: MODEL TRAINING & MLFLOW TRACKING")
    logger.info("=" * 60)

    mlflow.set_experiment("bank-term-deposit-prediction")

    with mlflow.start_run():
        model_pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ])

        model_pipeline.fit(X_train, y_train)
        logger.info("Model training completed successfully")

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("dropped_leakage_features", ",".join(LEAKAGE_COLUMNS))

    # ============================================================
    # 6. MODEL EVALUATION
    # ============================================================
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 60)

        y_pred = model_pipeline.predict(X_test)
        y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logger.info("Performance Metrics")
        logger.info("Accuracy:  %.4f", accuracy)
        logger.info("Precision: %.4f", precision)
        logger.info("Recall:    %.4f", recall)
        logger.info("F1-Score:  %.4f", f1)
        logger.info("ROC-AUC:   %.4f", roc_auc)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        logger.info("Classification Report:\n%s", classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))

        cm = confusion_matrix(y_test, y_pred)
        logger.info("Confusion Matrix")
        logger.info("                 Predicted No  Predicted Yes")
        logger.info("  Actual No      %10s  %13s", cm[0][0], cm[0][1])
        logger.info("  Actual Yes     %10s  %13s", cm[1][0], cm[1][1])

        feature_names = numeric_features + list(
            model_pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .get_feature_names_out(categorical_features)
        )
        importances = model_pipeline.named_steps["classifier"].feature_importances_
        feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
        feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)
        logger.info("Top 15 Feature Importances")
        for _, row in feat_imp.iterrows():
            logger.info("  %-30s %.4f", row["feature"], row["importance"])

    # ============================================================
    # 7. SAVE MODEL
    # ============================================================
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 7: SAVING MODEL")
        logger.info("=" * 60)

        joblib.dump(model_pipeline, MODEL_PATH)
        logger.info("Model saved as '%s'", MODEL_PATH)

        metadata = {
            "model_type": "RandomForestClassifier",
            "features": list(X.columns),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "dropped_leakage_features": LEAKAGE_COLUMNS,
            "target_classes": list(le.classes_),
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
            },
            "training_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
        }

        with METADATA_PATH.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Model metadata saved as '%s'", METADATA_PATH)

        columns_info = {
            "all_columns": list(X.columns),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "categorical_values": {},
        }
        for col in categorical_features:
            columns_info["categorical_values"][col] = list(df[col].unique())

        with COLUMNS_INFO_PATH.open("w", encoding="utf-8") as f:
            json.dump(columns_info, f, indent=2)
        logger.info("Column info saved as '%s'", COLUMNS_INFO_PATH)

        mlflow.sklearn.log_model(model_pipeline, "model")
        logger.info("Model logged to MLflow successfully")

    logger.info("\n%s", "=" * 60)
    logger.info("ALL STEPS COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("Files generated:")
    logger.info("  - %s (trained model pipeline)", MODEL_PATH)
    logger.info("  - %s (model performance info)", METADATA_PATH)
    logger.info("  - %s (feature info for API)", COLUMNS_INFO_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training pipeline failed")
        raise
