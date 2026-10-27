import pytest

from ai4i_mlops.config import MODEL_PATH
from ai4i_mlops.train import main as train_main


# Si no hay modelo entrenado, se entrena una vez antes de todas las pruebas
@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train_main()
