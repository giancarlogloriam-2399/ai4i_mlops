import os

import gradio as gr
import requests

from ai4i_mlops.config import DATASET_RANGES, TYPE_VALUES

# La interfaz NO carga el modelo: le pregunta a la API por HTTP
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 60


def predecir_falla(type, air_temperature, process_temperature, rotational_speed, torque, tool_wear):
    sample = {
        "type": type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=sample, timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        raise gr.Error(f"No se pudo conectar a la API en {API_URL}. Verifica que esté levantada.")

    if not response.ok:
        raise gr.Error(f"La API respondió con error {response.status_code}: {response.text}")

    result = response.json()
    probabilidad = result["failure_probability"]
    probabilidades = {
        "Falla": probabilidad,
        "Sin falla": 1 - probabilidad
    }
    resumen = (
        f"Predicción: {result['label']}\n"
        f"Probabilidad de falla: {probabilidad:.1%}\n"
        f"Nivel de riesgo: {result['risk_level'].upper()}"
    )
    return probabilidades, resumen


def crear_slider(columna: str, valor: float, paso: float, etiqueta: str) -> gr.Slider:
    # Los límites del slider son los valores reales del dataset
    minimo, maximo = DATASET_RANGES[columna]
    return gr.Slider(minimo, maximo, value=valor, step=paso, label=etiqueta)


demo = gr.Interface(
    fn=predecir_falla,
    inputs=[
        gr.Dropdown(TYPE_VALUES, value="M", label="Calidad del producto (Type)"),
        crear_slider("air_temperature", 300.0, 0.1, "Temperatura del aire (K)"),
        crear_slider("process_temperature", 310.0, 0.1, "Temperatura del proceso (K)"),
        crear_slider("rotational_speed", 1500, 1, "Velocidad de rotación (rpm)"),
        crear_slider("torque", 40.0, 0.1, "Torque (Nm)"),
        crear_slider("tool_wear", 100, 1, "Desgaste de la herramienta (min)"),
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilidad"),
        gr.Textbox(label="Resultado para el planner de mantenimiento"),
    ],
    examples=[
        ["M", 298.1, 308.6, 1551, 42.8, 0],      # máquina sana
        ["L", 298.9, 309.1, 1410, 65.7, 191],    # desgaste y torque altos
        ["L", 303.5, 311.9, 1330, 52.0, 100],    # sobrecalentamiento
    ],
    title="🔧 Mantenimiento predictivo AI4I 2020",
    description="Ingresa las condiciones de operación de la máquina para estimar el riesgo de falla",
    flagging_mode="never"
)


if __name__ == "__main__":
    demo.launch()
