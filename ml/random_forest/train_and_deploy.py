import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "failure_within_7d"


def build_pipeline(categorical_features, numeric_features):
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def load_dataset(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    return df


def train_and_save(data_path, output_dir, test_size):
    df = load_dataset(data_path)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(categorical_features, numeric_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "random_forest_failure_predictor.joblib"
    metrics_path = output_dir / "random_forest_metrics.json"

    joblib.dump(pipeline, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and deploy a RandomForest model for robot failure prediction."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Data") / "industrial_robot_sensor_data.txt",
        help="Path to the sensor data file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml_models"),
        help="Directory where trained model artifacts will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_and_save(args.data_path, args.output_dir, args.test_size)


if __name__ == "__main__":
    main()
