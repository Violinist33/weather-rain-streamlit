"""Навчання: python train.py --data /path/to/weatherAUS.csv."""
from pathlib import Path
import argparse
import hashlib
import json

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent


def train(data_path):
    raw = pd.read_csv(data_path, parse_dates=["Date"])
    data = raw.dropna(subset=["RainTomorrow"]).copy()
    # Раніші спостереження для навчання, пізніші — для перевірки.
    train_df = data[data.Date.dt.year <= 2014].copy()
    valid_df = data[data.Date.dt.year == 2015].copy()
    test_df = data[data.Date.dt.year >= 2016].copy()
    # RISK_MM (якщо є в іншій версії CSV) містить опади ЗАВТРА: це витік.
    input_cols = [c for c in data if c not in ["Date", "RainTomorrow", "RISK_MM"]]
    numeric_cols = train_df[input_cols].select_dtypes("number").columns.tolist()
    categorical_cols = [c for c in input_cols if c not in numeric_cols]

    imputer = SimpleImputer(strategy="mean").fit(train_df[numeric_cols])
    imputed = pd.DataFrame(imputer.transform(train_df[numeric_cols]), columns=numeric_cols)
    scaler = MinMaxScaler().fit(imputed)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    def transform(frame):
        numbers = pd.DataFrame(imputer.transform(frame[numeric_cols]), columns=numeric_cols)
        numbers = pd.DataFrame(scaler.transform(numbers), columns=numeric_cols)
        categories = pd.DataFrame(encoder.transform(frame[categorical_cols]), columns=encoded_cols)
        return pd.concat([numbers, categories], axis=1)

    X_train = transform(train_df)
    model = RandomForestClassifier(n_estimators=150, max_depth=16,
                                   min_samples_leaf=5, random_state=42, n_jobs=2)
    print(f"Навчання Random Forest: {len(train_df)} рядків, {X_train.shape[1]} ознак", flush=True)
    model.fit(X_train, train_df.RainTomorrow)
    metrics = {}
    for name, frame in [("validation", valid_df), ("test", test_df)]:
        X = transform(frame)
        y = frame.RainTomorrow
        predictions = model.predict(X)
        p = model.predict_proba(X)[:, list(model.classes_).index("Yes")]
        metrics[name] = {
            "rows": len(frame), "start": str(frame.Date.min().date()), "end": str(frame.Date.max().date()),
            "accuracy": accuracy_score(y, predictions),
            "precision_yes": precision_score(y, predictions, pos_label="Yes", zero_division=0),
            "recall_yes": recall_score(y, predictions, pos_label="Yes"),
            "f1_yes": f1_score(y, predictions, pos_label="Yes"),
            "roc_auc": roc_auc_score(y == "Yes", p),
            "baseline_always_no_accuracy": float((y == "No").mean()),
            "confusion_matrix_labels": ["No", "Yes"],
            "confusion_matrix": confusion_matrix(y, predictions, labels=["No", "Yes"]).tolist(),
        }
    bundle = dict(model=model, imputer=imputer, scaler=scaler, encoder=encoder,
                  input_cols=input_cols, target_col="RainTomorrow", numeric_cols=numeric_cols,
                  categorical_cols=categorical_cols, encoded_cols=encoded_cols)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models/aussie_rain.joblib", compress=("zlib", 3))
    report = {
        "model": "RandomForestClassifier", "sklearn_version": sklearn.__version__,
        "parameters": model.get_params(), "data_rows": len(raw),
        "dropped_missing_target": len(raw) - len(data), "train_rows": len(train_df),
        "train_start": str(train_df.Date.min().date()), "train_end": str(train_df.Date.max().date()),
        "input_features": len(input_cols), "encoded_features": X_train.shape[1],
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "source": "https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package",
        "metrics": metrics,
    }
    (ROOT / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    train(parser.parse_args().data)
