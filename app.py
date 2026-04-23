from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).parent / "data" / "diabetes.csv"
TARGET_COLUMN = "Outcome"
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

FEATURE_LABELS = {
    "Pregnancies": "Gestações",
    "Glucose": "Glicose",
    "BloodPressure": "Pressão arterial",
    "SkinThickness": "Espessura da pele",
    "Insulin": "Insulina",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Histórico familiar",
    "Age": "Idade",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data[ZERO_AS_MISSING_COLUMNS] = data[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    return data


@st.cache_resource
def train_model(data: pd.DataFrame) -> tuple[Pipeline, pd.Series, float, float]:
    x = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
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
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    importances = pd.Series(
        model.named_steps["classifier"].feature_importances_,
        index=x.columns,
    ).sort_values(ascending=True)

    return model, importances, accuracy, roc_auc


def build_sidebar_inputs(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Dados do paciente")

    values = {}
    for column in data.drop(columns=TARGET_COLUMN).columns:
        series = data[column]
        min_value = float(series.min())
        max_value = float(series.max())
        median_value = float(series.median())

        if column in {"Pregnancies", "Age"}:
            values[column] = st.sidebar.number_input(
                FEATURE_LABELS[column],
                min_value=int(min_value),
                max_value=int(max_value),
                value=int(median_value),
                step=1,
            )
        else:
            values[column] = st.sidebar.slider(
                FEATURE_LABELS[column],
                min_value=min_value,
                max_value=max_value,
                value=median_value,
            )

    return pd.DataFrame([values])


def plot_feature_importance(importances: pd.Series) -> plt.Figure:
    labels = [FEATURE_LABELS.get(feature, feature) for feature in importances.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, importances.values, color="#2f80ed")
    ax.set_xlabel("Importância")
    ax.set_title("Importância das features")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Diabetes Predictor",
        page_icon="🩺",
        layout="wide",
    )

    data = load_data()
    model, importances, accuracy, roc_auc = train_model(data)
    patient_data = build_sidebar_inputs(data)

    probability = model.predict_proba(patient_data)[0, 1]
    prediction = int(probability >= 0.5)

    st.title("Diabetes Predictor App")
    st.caption("Modelo Random Forest treinado com o Pima Indians Diabetes Dataset.")

    metric_label = "Risco elevado" if prediction else "Risco baixo"
    metric_delta = f"{probability:.1%} de probabilidade estimada"

    result_col, accuracy_col, auc_col = st.columns(3)
    result_col.metric("Predição", metric_label, metric_delta)
    accuracy_col.metric("Acurácia no teste", f"{accuracy:.1%}")
    auc_col.metric("ROC AUC no teste", f"{roc_auc:.1%}")

    st.divider()

    form_col, chart_col = st.columns([0.9, 1.1])

    with form_col:
        st.subheader("Entrada usada na predição")
        st.dataframe(
            patient_data.rename(columns=FEATURE_LABELS),
            hide_index=True,
            use_container_width=True,
        )

        if prediction:
            st.warning(
                "O modelo encontrou padrão compatível com maior risco. Use este resultado apenas como apoio educacional."
            )
        else:
            st.success(
                "O modelo encontrou padrão compatível com menor risco. Use este resultado apenas como apoio educacional."
            )

    with chart_col:
        st.subheader("Importância das features")
        st.pyplot(plot_feature_importance(importances), use_container_width=True)

    with st.expander("Amostra do dataset"):
        st.dataframe(data.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
