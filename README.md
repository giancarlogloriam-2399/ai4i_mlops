# Mantenimiento predictivo con AI4I 2020

Proyecto final del curso Data Science G8 (Maestría UNI – TECSUP). Lleva un modelo de Machine Learning de punta a punta:

```
datos → entrenamiento → API → interfaz web → contenedor → CI/CD y nube
```

La estructura sigue el proyecto de referencia del curso ([iris_mlops](https://github.com/marcell-llerena/iris_mlops)), adaptado a un problema distinto: anticipar fallas en máquinas industriales.

## El problema

En una planta, una falla no prevista detiene la línea y obliga a reparar de urgencia. El **planner de mantenimiento** necesita saber qué máquinas tienen más riesgo para programar una intervención antes de que se rompan.

El modelo recibe las condiciones de operación de una máquina y estima la probabilidad de que falle (`machine_failure` = 1).

## El dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI). Son 10 000 registros sintéticos que imitan las condiciones de operación de una máquina industrial con una herramienta que se desgasta. El CSV está en `data/raw/ai4i2020.csv` y forma parte del repositorio, así que el CI y Docker no dependen de que el sitio de UCI esté disponible.

| Variable | Descripción | Rango en el dataset |
|---|---|---|
| `type` | Calidad del producto: L (baja), M (media), H (alta) | L 60 %, M 30 %, H 10 % |
| `air_temperature` | Temperatura del aire (K) | 295.3 – 304.5 |
| `process_temperature` | Temperatura del proceso (K) | 305.7 – 313.8 |
| `rotational_speed` | Velocidad de rotación (rpm) | 1168 – 2886 |
| `torque` | Torque (Nm) | 3.8 – 76.6 |
| `tool_wear` | Desgaste de la herramienta (min) | 0 – 253 |
| `machine_failure` | **Target**: 1 si la máquina falló | 339 fallas (3.39 %) |

Las columnas originales se renombran a snake_case y sin corchetes (`Air temperature [K]` pasa a ser `air_temperature`). Los porcentajes de `type` salen del CSV; la ficha de UCI menciona 50/30/20, pero el archivo publicado tiene 60/30/10.

### Columnas eliminadas y por qué hay leakage

- `UDI` y `Product ID` son identificadores. No describen la máquina y el modelo solo aprendería a memorizarlos.
- `TWF`, `HDF`, `PWF`, `OSF` y `RNF` indican el **tipo** de falla: desgaste de herramienta, disipación de calor, potencia, sobreesfuerzo y falla aleatoria. Se registran después de que la máquina falla y explican casi todas las fallas: en el CSV, siempre que `TWF`, `HDF`, `PWF` u `OSF` vale 1, `machine_failure` también vale 1. Dejarlas sería **data leakage**: el modelo tendría la respuesta en la entrada, sacaría métricas casi perfectas en el test y no serviría en producción, donde esa información todavía no existe al momento de predecir.

## Por qué F1 y recall, y no accuracy

Solo el 3.4 % de los registros son fallas. Un modelo que siempre responde "sin falla" tendría **96.6 % de accuracy** sin detectar ni una sola falla. Por eso la accuracy aparece solo como dato informativo.

Las métricas que importan se calculan sobre la clase 1 (falla):

- **Recall**: de las fallas reales, cuántas detecta el modelo. Es la prioridad, porque una falla no detectada significa una parada no planificada.
- **Precision**: de las alertas que da el modelo, cuántas son fallas reales. Si es baja, el planner pierde tiempo en inspecciones innecesarias.
- **F1**: equilibrio entre las dos anteriores.
- **ROC-AUC**: qué tan bien ordena el modelo las máquinas por riesgo, sin depender del umbral.

Para compensar el desbalance, el Random Forest usa `class_weight="balanced"`.

## Resultados del modelo

Random Forest (200 árboles, `max_depth=15`, `min_samples_leaf=3`, `class_weight="balanced"`), con split estratificado 80/20 y `random_state=42`. Las métricas son sobre el conjunto de test (2000 registros, 68 fallas):

| Métrica (clase falla) | Valor | Mínimo exigido |
|---|---|---|
| F1 | **0.640** | 0.59 |
| Recall | **0.824** | 0.77 |
| Precision | 0.523 | – |
| ROC-AUC | 0.972 | – |
| Accuracy (informativa) | 0.969 | – |

Matriz de confusión:

|  | Predice sin falla | Predice falla |
|---|---|---|
| **Real sin falla** | 1881 | 51 |
| **Real falla** | 12 | 56 |

El modelo detecta 56 de las 68 fallas. A cambio genera 51 falsas alarmas sobre 1932 máquinas sanas, lo que para mantenimiento es un costo razonable: es más barata una inspección de más que una parada imprevista.

Los hiperparámetros se eligieron comparando algunas configuraciones con validación cruzada de 5 folds **solo sobre el conjunto de entrenamiento**. Un árbol sin límite de profundidad conseguía un F1 un poco mayor (≈0.66), pero con recall de ≈0.62. Se prefirió la configuración con mejor recall.

### Puerta de calidad

`train.py` compara las métricas con `MIN_F1 = 0.59` y `MIN_RECALL = 0.77`, definidos en `config.py`. Esos valores están unos 0.05 por debajo del resultado obtenido. Si el modelo no los alcanza, el entrenamiento lanza un error y **no guarda el modelo**. Como el Dockerfile entrena dentro de la imagen, un modelo malo tampoco llega a construirse ni a desplegarse.

Cada entrenamiento exitoso guarda `models/model.joblib` y `models/metrics.json` (versión, fecha, tipo de modelo, hiperparámetros, features, métricas y matriz de confusión).

## Arquitectura

```
                 data/raw/ai4i2020.csv
                          │
                          ▼
   data.py ──── limpia, valida y separa train/test (estratificado)
                          │
                          ▼
   train.py ─── Pipeline: OneHotEncoder(type) + RandomForest
                │  puerta de calidad (F1 y recall)
                ▼
        models/model.joblib + models/metrics.json
                          │
                          ▼
   predict.py ─ carga el modelo una sola vez (caché)
                          │
                          ▼
   api.py (FastAPI :8000)  GET /   GET /model-info   POST /predict
                          ▲
                          │ HTTP (API_URL)
   app/gradio_app.py (Gradio :7860) ── planner de mantenimiento

   GitHub push ─► CI (datos → entrenar → pytest) ─► CD (deploy hooks) ─► Render
```

```
ai4i_mlops/
├── .github/workflows/      ci.yml y cd.yml
├── app/gradio_app.py       interfaz web (consume la API)
├── data/raw/ai4i2020.csv   dataset original
├── models/                 se genera al entrenar (no se sube a git)
├── src/ai4i_mlops/
│   ├── config.py           rutas, features, rangos, hiperparámetros y umbrales
│   ├── data.py             carga, limpieza, validación y split
│   ├── train.py            entrenamiento, métricas y puerta de calidad
│   ├── predict.py          predicción con el modelo cacheado
│   └── api.py              API FastAPI
├── tests/                  pruebas de datos, modelo y API
├── Dockerfile, compose.yaml, .dockerignore
└── pyproject.toml, uv.lock, .python-version
```

## Cómo ejecutarlo desde cero

Requisitos: git y conexión a internet. uv se encarga de instalar Python 3.14 si hace falta.

**1. Instalar uv**

Windows (PowerShell):

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux / macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clonar el repositorio e instalar las dependencias**

```bash
git clone <URL-del-repositorio> ai4i_mlops
```

```bash
cd ai4i_mlops
```

```bash
uv sync
```

**3. Preparar y validar los datos**

```bash
uv run preparar-datos
```

Imprime filas, columnas y conteo de fallas. Si el CSV tiene nulos, valores fuera de rango o un `type` distinto de L/M/H, se detiene con un error.

**4. Entrenar el modelo**

```bash
uv run entrenar
```

Para probar una predicción desde la terminal:

```bash
uv run predecir
```

**5. Levantar la API** (deja esta terminal abierta)

```bash
uv run fastapi run src/ai4i_mlops/api.py --port 8000
```

La documentación interactiva queda en http://localhost:8000/docs, con un ejemplo ya cargado en `/predict`.

**6. Levantar la interfaz Gradio** (en otra terminal)

```bash
uv run python app/gradio_app.py
```

Se abre en http://localhost:7860. Por defecto busca la API en `http://127.0.0.1:8000`; se puede cambiar con la variable de entorno `API_URL`.

**7. Correr las pruebas**

```bash
uv run pytest -v
```

**8. Todo con Docker** (requiere Docker Desktop)

```bash
docker compose up --build
```

Levanta la API en el puerto 8000 y Gradio en el 7860. El modelo se entrena durante el build.

## Endpoints de la API

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/` | Health check: `{"status": "ok", ...}` |
| GET | `/model-info` | Contenido de `metrics.json`. 503 si no hay modelo entrenado |
| POST | `/predict` | Predicción, probabilidad de falla y nivel de riesgo. 422 si la entrada no es válida, 503 si no hay modelo |

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"type\":\"L\",\"air_temperature\":298.9,\"process_temperature\":309.1,\"rotational_speed\":1410,\"torque\":65.7,\"tool_wear\":191}"
```

Respuesta:

```json
{"prediction": 1, "label": "Falla", "failure_probability": 0.9651, "risk_level": "alto"}
```

- `prediction`: 1 si la probabilidad es ≥ 0.5 (`DECISION_THRESHOLD`).
- `risk_level`: `bajo` (< 0.3), `medio` (0.3 – 0.6) o `alto` (≥ 0.6). Los cortes están en `config.py`.

La API valida cada campo con Pydantic: `type` solo acepta `L`, `M` o `H`, y las variables numéricas deben estar dentro de los rangos físicos de `config.py` (por ejemplo, torque entre 0 y 100 Nm). Si `type` es `"X"` o el torque es negativo, la respuesta es **422**.

## Pruebas

12 pruebas en tres capas:

- **Datos** (`test_data.py`): 10 000 filas, sin columnas de leakage, `type` solo L/M/H, y `validate_data` rechaza nulos y valores fuera de rango.
- **Modelo** (`test_model.py`): supera la puerta de calidad y `predict()` devuelve las claves correctas con una probabilidad entre 0 y 1.
- **API** (`test_api.py`): health 200, `/predict` válido 200, `type` inválido 422, torque negativo 422 y `/model-info` 200.

## CI/CD

- **CI** (`.github/workflows/ci.yml`): en cada push a `main` instala dependencias con `uv sync --locked`, prepara los datos, entrena (con puerta de calidad), corre `pytest -v` y sube la carpeta `models/` como artifact.
- **CD** (`.github/workflows/cd.yml`): se ejecuta cuando termina el CI y solo si terminó en verde. Llama a los deploy hooks de Render guardados como secrets de GitHub (`RENDER_DEPLOY_API` y `RENDER_DEPLOY_UI`). Ninguna URL ni clave está escrita en el código.

## Despliegue en Render

Estos pasos se hacen a mano una sola vez.

1. **Subir el repositorio a GitHub** (rama `main`) y verificar en la pestaña *Actions* que el CI termine en verde.
2. **Crear el servicio de la API** en Render: *New → Web Service*, conectar el repositorio y elegir *Docker* como entorno. No hace falta cambiar el comando, porque el Dockerfile ya usa `${PORT:-8000}`. En *Settings*, dejar **Auto-Deploy en "No"**, para que solo despliegue el CD después de un CI en verde.
3. **Copiar la URL pública de la API** (por ejemplo `https://ai4i-api.onrender.com`) y probarla abriendo `/docs`.
4. **Crear el servicio de la interfaz**: otro *Web Service* con el mismo repositorio y entorno Docker. En *Docker Command* poner `python app/gradio_app.py` y agregar estas variables de entorno:
   - `API_URL` = URL de la API del paso 3
   - `GRADIO_SERVER_NAME` = `0.0.0.0`
   - `GRADIO_SERVER_PORT` = `7860`
   - `PORT` = `7860`

   También aquí, Auto-Deploy en "No".
5. **Copiar los deploy hooks**: en cada servicio, *Settings → Deploy Hook*.
6. **Guardarlos como secrets en GitHub**: *Settings → Secrets and variables → Actions → New repository secret*:
   - `RENDER_DEPLOY_API` = deploy hook de la API
   - `RENDER_DEPLOY_UI` = deploy hook de la interfaz
7. **Probar la cadena completa**: hacer un push a `main`. Primero corre el CI y, si pasa, el CD dispara los dos despliegues.

En el plan gratuito de Render los servicios se duermen tras un rato sin uso. La primera petición puede tardar cerca de un minuto, y por eso la interfaz espera hasta 60 segundos a la API.

## Limitaciones

- **Datos sintéticos.** AI4I 2020 es una simulación. Las fallas siguen reglas bastante limpias (por ejemplo, la falla por calor depende de la diferencia de temperaturas y de la velocidad), algo que con sensores reales no pasa. Las métricas no se trasladan tal cual a una planta real.
- **RNF no se puede predecir y las etiquetas tienen ruido.** Según UCI, la falla aleatoria (RNF) ocurre con 0.1 % de probabilidad sin importar las condiciones de operación, así que ningún modelo basado en estas variables puede anticiparla. Además, el CSV no es del todo consistente: 18 de las 19 filas con `RNF` = 1 tienen `machine_failure` = 0, y 9 fallas no tienen ningún tipo de falla marcado. Esas fallas sin causa visible limitan el recall máximo posible.
- **El umbral de 0.5 es ajustable.** Bajarlo aumenta el recall a costa de más falsas alarmas; subirlo hace lo contrario. El valor correcto depende de cuánto le cuesta a la planta una parada frente a una inspección, y se cambia en `DECISION_THRESHOLD` de `config.py`.
- **Una sola lectura por máquina.** El modelo no usa historia temporal, así que no ve tendencias como un torque que va subiendo día a día.

## Mejoras futuras

- **MLflow** para registrar cada experimento (hiperparámetros, métricas y modelo) y comparar versiones.
- **Monitoreo de drift**: comparar la distribución de las peticiones que llegan a la API con la de entrenamiento y avisar cuando se alejen.
- **Reentrenamiento** periódico o disparado por drift, que pase por la misma puerta de calidad antes de reemplazar al modelo en producción.
- Ajustar el umbral de decisión con un análisis de costos junto al área de mantenimiento.
