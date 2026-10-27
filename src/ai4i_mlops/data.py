import pandas as pd
from sklearn.model_selection import train_test_split

from ai4i_mlops.config import (
    DATA_PATH,
    DROP_COLUMNS,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RENAME_COLUMNS,
    TARGET,
    TARGET_VALUES,
    TEST_SIZE,
    TYPE_VALUES,
    VALID_RANGES,
)


def read_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {DATA_PATH}")
    # utf-8-sig: el CSV de UCI trae un carácter BOM al inicio
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Quitar identificadores y columnas de leakage, luego renombrar
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.rename(columns=RENAME_COLUMNS)
    return df


def validate_data(df: pd.DataFrame) -> None:
    columnas_perdidas = set(FEATURES + [TARGET]) - set(df.columns)
    if columnas_perdidas:
        raise ValueError(f"Faltan columnas: {columnas_perdidas}")

    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Hay valores nulos en los datos")

    tipos_invalidos = set(df["type"]) - set(TYPE_VALUES)
    if tipos_invalidos:
        raise ValueError(f"Valores de type no permitidos: {tipos_invalidos}")

    for columna in NUMERIC_FEATURES:
        minimo, maximo = VALID_RANGES[columna]
        fuera_de_rango = (df[columna] < minimo) | (df[columna] > maximo)
        if fuera_de_rango.any():
            raise ValueError(
                f"{columna} tiene valores fuera de rango [{minimo}, {maximo}]"
            )

    targets_invalidos = set(df[TARGET]) - set(TARGET_VALUES)
    if targets_invalidos:
        raise ValueError(f"El target solo puede ser 0 o 1: {targets_invalidos}")


def load_data() -> pd.DataFrame:
    df = clean_data(read_raw_data())
    validate_data(df)
    return df


def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def main() -> None:
    df = load_data()
    print(f"Datos validados: {DATA_PATH}")
    print(f"Filas: {len(df)} | Columnas: {list(df.columns)}")
    print("Cantidad de registros por clase (1 = falla):")
    print(df[TARGET].value_counts().to_string())
    porcentaje = df[TARGET].mean() * 100
    print(f"Porcentaje de fallas: {porcentaje:.2f} %")


if __name__ == "__main__":
    main()
