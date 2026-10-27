from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai4i_mlops import __version__
from ai4i_mlops.config import VALID_RANGES
from ai4i_mlops.predict import load_model_info, predict


app = FastAPI(
    title="AI4I Predictive Maintenance API",
    description="API para predecir si una máquina va a fallar a partir de sus condiciones de operación",
    version=__version__
)


class MachineInput(BaseModel):
    # Los rangos válidos vienen de config.py (los mismos que usa data.py)
    type: Literal["L", "M", "H"] = Field(description="Calidad del producto: L (baja), M (media), H (alta)")
    air_temperature: float = Field(
        ge=VALID_RANGES["air_temperature"][0],
        le=VALID_RANGES["air_temperature"][1],
        description="Temperatura del aire (K)"
    )
    process_temperature: float = Field(
        ge=VALID_RANGES["process_temperature"][0],
        le=VALID_RANGES["process_temperature"][1],
        description="Temperatura del proceso (K)"
    )
    rotational_speed: float = Field(
        ge=VALID_RANGES["rotational_speed"][0],
        le=VALID_RANGES["rotational_speed"][1],
        description="Velocidad de rotación (rpm)"
    )
    torque: float = Field(
        ge=VALID_RANGES["torque"][0],
        le=VALID_RANGES["torque"][1],
        description="Torque (Nm)"
    )
    tool_wear: float = Field(
        ge=VALID_RANGES["tool_wear"][0],
        le=VALID_RANGES["tool_wear"][1],
        description="Desgaste de la herramienta (min)"
    )

    # Ejemplo que aparece precargado en /docs
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "L",
                    "air_temperature": 298.9,
                    "process_temperature": 309.1,
                    "rotational_speed": 1410,
                    "torque": 65.7,
                    "tool_wear": 191
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    prediction: int
    label: str
    failure_probability: float
    risk_level: str


# Endpoints
@app.get("/")
def health():
    return {"status": "ok", "message": "AI4I API operativa"}

@app.get("/model-info")
def model_info():
    try:
        return load_model_info()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado")

@app.post("/predict", response_model=PredictionOutput)
def predict_failure(sample: MachineInput):
    try:
        return predict(**sample.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado")
