from typing import Any
from datetime import datetime
import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ai4i_mlops import __version__
from ai4i_mlops.config import (
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    FEATURES,
    METRICS_PATH,
    MIN_F1,
    MIN_RECALL,
    MODEL_PARAMS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TYPE_VALUES,
)
from ai4i_mlops.data import load_data, split_data


def build_pipeline() -> Pipeline:
    # Type pasa a columnas 0/1; las numéricas se dejan igual
    # (un Random Forest no necesita escalar)
    preprocessor = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(categories=[TYPE_VALUES]), CATEGORICAL_FEATURES),
            ("numericas", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocessor),
            ("modelo", RandomForestClassifier(**MODEL_PARAMS)),
        ]
    )


def train_model(X_train, y_train) -> Pipeline:
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    # Probabilidad de la clase 1 (falla) y predicción con el umbral
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    return {
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def check_quality(metrics: dict[str, Any]) -> None:
    # Puerta de calidad: si no se cumple, el modelo NO se guarda
    if metrics["f1"] < MIN_F1:
        raise ValueError(f"F1 = {metrics['f1']} es menor al mínimo {MIN_F1}. No se guardará")
    if metrics["recall"] < MIN_RECALL:
        raise ValueError(f"Recall = {metrics['recall']} es menor al mínimo {MIN_RECALL}. No se guardará")


def save_model(model, metrics: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    model_metadata = {
        "model_version": __version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": type(model.named_steps["modelo"]).__name__,
        "params": MODEL_PARAMS,
        "features": FEATURES,
        "decision_threshold": DECISION_THRESHOLD,
        "metrics": {
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "roc_auc": metrics["roc_auc"],
            "accuracy": metrics["accuracy"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
    }

    METRICS_PATH.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")


def main() -> None:
    print("1) Cargando datos...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"    Entrenamiento: {len(X_train)} filas | Test: {len(X_test)} filas")

    print("2) Entrenando modelo...")
    model = train_model(X_train, y_train)

    print("3) Evaluando modelo (clase 1 = falla)...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"    Accuracy:  {metrics['accuracy']:.4f} (solo informativa)")
    print(f"    Matriz de confusión [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    print("4) Revisando la puerta de calidad...")
    check_quality(metrics)
    print(f"    OK: F1 >= {MIN_F1} y Recall >= {MIN_RECALL}")

    print("5) Guardando el modelo...")
    save_model(model, metrics)
    print(f"    Modelo: {MODEL_PATH}")
    print(f"    Métricas: {METRICS_PATH}")
    print("Listo!")


if __name__ == "__main__":
    main()
