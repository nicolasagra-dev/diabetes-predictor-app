from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
    "Pregnancies": "Gestacoes",
    "Glucose": "Glicose",
    "BloodPressure": "Pressao arterial",
    "SkinThickness": "Espessura da pele",
    "Insulin": "Insulina",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Historico familiar",
    "Age": "Idade",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data[ZERO_AS_MISSING_COLUMNS] = data[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    return data


@st.cache_resource
def train_model(
    data: pd.DataFrame,
) -> tuple[Pipeline, pd.Series, dict[str, float], pd.DataFrame]:
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
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
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

    return model, importances, metrics, evaluation


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
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia das features")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(evaluation: pd.DataFrame) -> plt.Figure:
    matrix = confusion_matrix(evaluation["real"], evaluation["predicao"])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Baixo risco", "Alto risco"])
    ax.set_yticks([0, 1], labels=["Baixo risco", "Alto risco"])
    ax.set_xlabel("Predicao")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusao")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_feature_distribution(data: pd.DataFrame, feature: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome, color, label in [(0, "#27ae60", "Baixo risco"), (1, "#eb5757", "Alto risco")]:
        ax.hist(
            data.loc[data[TARGET_COLUMN] == outcome, feature].dropna(),
            bins=24,
            alpha=0.7,
            color=color,
            label=label,
        )
    ax.set_title(f"Distribuicao de {FEATURE_LABELS.get(feature, feature)}")
    ax.set_xlabel(FEATURE_LABELS.get(feature, feature))
    ax.set_ylabel("Pacientes")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Diabetes Predictor",
        page_icon="DP",
        layout="wide",
    )

    data = load_data()
    model, importances, metrics, evaluation = train_model(data)
    patient_data = build_sidebar_inputs(data)

    probability = model.predict_proba(patient_data)[0, 1]
    prediction = int(probability >= 0.5)

    st.title("Diabetes Predictor App")
    st.caption("Modelo Random Forest treinado com o Pima Indians Diabetes Dataset.")

    metric_label = "Risco elevado" if prediction else "Risco baixo"
    metric_delta = f"{probability:.1%} de probabilidade estimada"

    result_col, accuracy_col, auc_col = st.columns(3)
    result_col.metric("Predicao", metric_label, metric_delta)
    accuracy_col.metric("Acuracia no teste", f"{metrics['accuracy']:.1%}")
    auc_col.metric("ROC AUC no teste", f"{metrics['roc_auc']:.1%}")

    st.divider()

    prediction_tab, metrics_tab, data_tab = st.tabs(["Predicao", "Metricas", "Dados"])

    with prediction_tab:
        form_col, chart_col = st.columns([0.9, 1.1])

        with form_col:
            st.subheader("Entrada usada na predicao")
            st.dataframe(
                patient_data.rename(columns=FEATURE_LABELS),
                hide_index=True,
                use_container_width=True,
            )

            st.progress(
                int(probability * 100),
                text=f"Probabilidade estimada: {probability:.1%}",
            )

            if prediction:
                st.warning(
                    "O modelo encontrou padrao compativel com maior risco. Use este resultado apenas como apoio educacional."
                )
            else:
                st.success(
                    "O modelo encontrou padrao compativel com menor risco. Use este resultado apenas como apoio educacional."
                )

        with chart_col:
            st.subheader("Importancia das features")
            st.pyplot(plot_feature_importance(importances), use_container_width=True)

    with metrics_tab:
        metric_columns = st.columns(5)
        metric_columns[0].metric("Acuracia", f"{metrics['accuracy']:.1%}")
        metric_columns[1].metric("Precisao", f"{metrics['precision']:.1%}")
        metric_columns[2].metric("Recall", f"{metrics['recall']:.1%}")
        metric_columns[3].metric("F1-score", f"{metrics['f1']:.1%}")
        metric_columns[4].metric("ROC AUC", f"{metrics['roc_auc']:.1%}")

        matrix_col, table_col = st.columns([0.9, 1.1])
        with matrix_col:
            st.pyplot(plot_confusion_matrix(evaluation), use_container_width=True)
        with table_col:
            st.subheader("Amostra das predicoes de teste")
            formatted_evaluation = evaluation.copy()
            formatted_evaluation["probabilidade"] = formatted_evaluation[
                "probabilidade"
            ].map(lambda value: f"{value:.1%}")
            st.dataframe(formatted_evaluation.head(20), use_container_width=True)

    with data_tab:
        overview_col, chart_col = st.columns([0.9, 1.1])
        with overview_col:
            st.subheader("Resumo do dataset")
            st.metric("Registros", f"{len(data):,}".replace(",", "."))
            st.metric("Features", data.drop(columns=TARGET_COLUMN).shape[1])
            st.metric("Casos positivos", f"{data[TARGET_COLUMN].mean():.1%}")
            st.dataframe(data.describe().T, use_container_width=True)

        with chart_col:
            feature = st.selectbox(
                "Feature para distribuicao",
                list(data.drop(columns=TARGET_COLUMN).columns),
                format_func=lambda value: FEATURE_LABELS.get(value, value),
            )
            st.pyplot(plot_feature_distribution(data, feature), use_container_width=True)

        with st.expander("Amostra do dataset"):
            st.dataframe(data.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
