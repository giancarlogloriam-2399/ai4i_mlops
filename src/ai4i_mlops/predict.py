from typing import Any
import json

import joblib
import pandas as pd

from ai4i_mlops.config import (
    DECISION_THRESHOLD,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    RISK_HIGH,
    RISK_MEDIUM,
)

# El modelo se guarda aquí la primera vez que se carga (caché)
_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def load_model_info() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {METRICS_PATH}")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_risk_level(probability: float) -> str:
    if probability >= RISK_HIGH:
        return "alto"
    if probability >= RISK_MEDIUM:
        return "medio"
    return "bajo"


def predict(
    type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float
) -> dict[str, Any]:
    model = load_model()
    sample = pd.DataFrame(
        [[type, air_temperature, process_temperature, rotational_speed, torque, tool_wear]],
        columns=FEATURES
    )
    # Columna 1 = probabilidad de falla
    probability = float(model.predict_proba(sample)[0][1])
    prediction = int(probability >= DECISION_THRESHOLD)

    if prediction == 1:
        label = "Falla"
    else:
        label = "Sin falla"

    return {
        "prediction": prediction,
        "label": label,
        "failure_probability": round(probability, 4),
        "risk_level": get_risk_level(probability),
    }


def main() -> None:
    example = {
        "type": "L",
        "air_temperature": 298.9,
        "process_temperature": 309.1,
        "rotational_speed": 1410,
        "torque": 65.7,
        "tool_wear": 191
    }
    print(f"Muestra ejemplo: {example}")
    print(f"Predicción: {predict(**example)}")


if __name__ == "__main__":
    main()
