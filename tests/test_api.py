from fastapi.testclient import TestClient

from ai4i_mlops.api import app


client = TestClient(app)

VALID_SAMPLE = {
    "type": "L",
    "air_temperature": 298.9,
    "process_temperature": 309.1,
    "rotational_speed": 1410,
    "torque": 65.7,
    "tool_wear": 191
}


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    response = client.post("/predict", json=VALID_SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"prediction", "label", "failure_probability", "risk_level"}
    assert 0 <= body["failure_probability"] <= 1


def test_predict_rejects_invalid_type():
    sample = dict(VALID_SAMPLE)
    sample["type"] = "X"
    response = client.post("/predict", json=sample)
    assert response.status_code == 422


def test_predict_rejects_negative_torque():
    sample = dict(VALID_SAMPLE)
    sample["torque"] = -5
    response = client.post("/predict", json=sample)
    assert response.status_code == 422


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "metrics" in response.json()
