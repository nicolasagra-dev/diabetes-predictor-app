from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).parent
DATA_PATH = ROOT_DIR / "data" / "diabetes.csv"
MODEL_PATH = ROOT_DIR / "models" / "diabetes_random_forest.joblib"
TARGET_COLUMN = "Outcome"
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

FEATURE_LABELS = {
    "Pregnancies": "Gestacoes",
    "Glucose": "Glicose",
    "BloodPressure": "Pressao arterial",
    "SkinThickness": "Espessura da pele",
    "Insulin": "Insulina",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Historico familiar",
    "Age": "Idade",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    data[ZERO_AS_MISSING_COLUMNS] = data[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=4,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model(data: pd.DataFrame) -> dict[str, object]:
    x = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    evaluation = pd.DataFrame(
        {
            "real": y_test.to_numpy(),
            "predicao": predictions,
            "probabilidade": probabilities,
        }
    )

    importances = pd.Series(
        model.named_steps["classifier"].feature_importances_,
        index=x.columns,
    ).sort_values(ascending=True)

    return {
        "model": model,
        "importances": importances,
        "metrics": metrics,
        "evaluation": evaluation,
        "feature_names": list(x.columns),
    }


def save_model(bundle: dict[str, object], path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: Path = MODEL_PATH) -> dict[str, object]:
    return joblib.load(path)

