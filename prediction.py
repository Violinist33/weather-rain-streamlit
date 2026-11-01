"""Підготовка одного спостереження та прогноз збереженою моделлю."""
import numpy as np
import pandas as pd


def preprocess(data, bundle):
    # Порядок ознак має бути таким самим, як під час навчання.
    frame = pd.DataFrame([data], columns=bundle["input_cols"])
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    numeric = frame[numeric_cols].astype(float)
    numeric = pd.DataFrame(
        bundle["imputer"].transform(numeric), columns=numeric_cols
    )
    numeric = pd.DataFrame(
        bundle["scaler"].transform(numeric), columns=numeric_cols
    )
    categories = frame[categorical_cols].astype(object)
    categories = categories.where(categories.notna(), np.nan)
    encoded = bundle["encoder"].transform(categories)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded = pd.DataFrame(encoded, columns=bundle["encoded_cols"])
    return pd.concat([numeric, encoded], axis=1)


def predict_weather(data, bundle):
    features = preprocess(data, bundle)
    model = bundle["model"]
    prediction = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    # Не припускаємо, що Yes завжди стоїть у другому стовпці.
    probability_by_class = dict(zip(model.classes_, probabilities))
    return {
        "prediction": prediction,
        "rain_probability": float(probability_by_class["Yes"]),
        "prediction_probability": float(probability_by_class[prediction]),
        "missing_count": sum(pd.isna(data.get(c)) for c in bundle["input_cols"]),
    }
