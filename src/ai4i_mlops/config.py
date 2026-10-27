from pathlib import Path

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "ai4i2020.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Nombres originales del CSV -> nombres en snake_case
RENAME_COLUMNS = {
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}

# Identificadores (no aportan información)
ID_COLUMNS = ["UDI", "Product ID"]

# Tipos de falla: dejarlos sería data leakage (revelan el target)
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

DROP_COLUMNS = ID_COLUMNS + LEAKAGE_COLUMNS

# Datos
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "machine_failure"

# Valores permitidos para la calidad del producto
TYPE_VALUES = ["L", "M", "H"]
TARGET_VALUES = [0, 1]

# Rangos físicos válidos (mínimo, máximo). Los usan data.py y la API.
# Son más amplios que los datos para aceptar lecturas razonables.
VALID_RANGES = {
    "air_temperature": (290.0, 310.0),      # K
    "process_temperature": (300.0, 320.0),  # K
    "rotational_speed": (1000.0, 3000.0),   # rpm
    "torque": (0.0, 100.0),                 # Nm
    "tool_wear": (0.0, 300.0),              # min
}

# Rangos reales observados en el dataset (los usa la interfaz Gradio)
DATASET_RANGES = {
    "air_temperature": (295.3, 304.5),
    "process_temperature": (305.7, 313.8),
    "rotational_speed": (1168.0, 2886.0),
    "torque": (3.8, 76.6),
    "tool_wear": (0.0, 253.0),
}

# Entrenamiento
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hiperparámetros
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Umbral de decisión: probabilidad desde la cual se predice "Falla"
DECISION_THRESHOLD = 0.5

# Niveles de riesgo según la probabilidad de falla
RISK_MEDIUM = 0.3   # desde aquí el riesgo es "medio"
RISK_HIGH = 0.6     # desde aquí el riesgo es "alto"

# Calidad mínima aceptable (clase 1 = falla)
MIN_F1 = 0.59
MIN_RECALL = 0.77
