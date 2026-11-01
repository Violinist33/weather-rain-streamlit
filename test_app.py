"""Перевірка прогнозів, пропусків і форми: python -m unittest -v."""
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

from prediction import predict_weather, preprocess

ROOT = Path(__file__).resolve().parent


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(ROOT / "models/aussie_rain.joblib")
        cls.example = json.loads((ROOT / "example.json").read_text())

    def test_features_and_missing_values(self):
        missing = dict(self.example, Sunshine=None, WindDir9am=None)
        before = joblib.hash(self.bundle)
        for values in [self.example, missing]:
            features = preprocess(values, self.bundle)
            self.assertEqual(list(features.columns), list(self.bundle["model"].feature_names_in_))
            self.assertTrue(np.isfinite(features.to_numpy()).all())
            result = predict_weather(values, self.bundle)
            self.assertIn(result["prediction"], ["Yes", "No"])
            self.assertTrue(0 <= result["rain_probability"] <= 1)
            expected = result["rain_probability"] if result["prediction"] == "Yes" else 1 - result["rain_probability"]
            self.assertAlmostEqual(expected, result["prediction_probability"])
        self.assertEqual(before, joblib.hash(self.bundle), "Прогноз не має перенавчати модель")

    def test_yes_and_no_probability_labels(self):
        dry = predict_weather(self.example, self.bundle)
        wet_input = dict(self.example, Humidity9am=95.0, Humidity3pm=98.0,
                         Rainfall=30.0, Sunshine=0.0, Cloud9am=8.0, Cloud3pm=8.0,
                         Pressure9am=995.0, Pressure3pm=990.0, RainToday="Yes")
        wet = predict_weather(wet_input, self.bundle)
        self.assertEqual(dry["prediction"], "No")
        self.assertEqual(wet["prediction"], "Yes")
        self.assertGreater(wet["rain_probability"], dry["rain_probability"])

    def test_form_example_and_validation(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.button[2].click().run()
        self.assertTrue(app.warning)
        app.button[0].click().run()
        app.button[2].click().run()
        self.assertFalse(app.exception)
        expected = predict_weather(self.example, self.bundle)
        self.assertEqual(app.metric[0].value, f"{expected['rain_probability']:.1%}")
        app.number_input(key="Sunshine").set_value(None)
        app.selectbox(key="WindDir9am").select("Не знаю")
        app.button[2].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["result"]["missing_count"], 2)
        app.number_input(key="MinTemp").set_value(40.0)
        app.number_input(key="MaxTemp").set_value(10.0)
        app.button[2].click().run()
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)
        app.button[1].click().run()
        self.assertIsNone(app.number_input(key="MinTemp").value)
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
