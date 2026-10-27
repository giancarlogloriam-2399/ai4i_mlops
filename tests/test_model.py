from ai4i_mlops.config import MIN_F1, MIN_RECALL
from ai4i_mlops.data import load_data, split_data
from ai4i_mlops.predict import predict
from ai4i_mlops.train import evaluate_model, train_model


def test_model_passes_quality_gate():
    X_train, X_test, y_train, y_test = split_data(load_data())
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["f1"] >= MIN_F1
    assert metrics["recall"] >= MIN_RECALL


def test_predict_output():
    result = predict(
        type="L",
        air_temperature=298.9,
        process_temperature=309.1,
        rotational_speed=1410,
        torque=65.7,
        tool_wear=191
    )
    assert set(result.keys()) == {"prediction", "label", "failure_probability", "risk_level"}
    assert result["prediction"] in [0, 1]
    assert 0 <= result["failure_probability"] <= 1
    assert result["risk_level"] in ["bajo", "medio", "alto"]
