"""Train the failure-risk model and a failure-pattern model.

Selection uses out-of-fold scores on the training split only. The holdout
split is scored once, after the model and the decision threshold are fixed.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    EXPLANATION_GROUPS,
    FAILURE_FLAGS,
    MODEL_FEATURES,
    PHYSICAL_MODES,
    build_features,
    failure_mode_label,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
RANDOM_STATE = 42
MIN_RECALL = 0.80
FORBIDDEN_FEATURES = {"TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure", "UDI", "Product ID"}


def load_raw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    required = {
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Unexpected AI4I file. Missing {sorted(missing)}.")
    return data


def candidates() -> dict:
    return {
        "Logistic regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=250,
            early_stopping=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def scores_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "false_alarms": int(fp),
        "missed_failures": int(fn),
        "caught_failures": int(tp),
        "true_healthy": int(tn),
    }


def operating_threshold(y_true: np.ndarray, scores: np.ndarray, min_recall: float = MIN_RECALL) -> float:
    """Catch at least `min_recall` of training-fold failures, then cut false alarms."""
    y_true = np.asarray(y_true)
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 401)))
    evaluated = []
    for threshold in candidates:
        predicted = scores >= threshold
        caught = np.sum(predicted & (y_true == 1))
        missed = np.sum(~predicted & (y_true == 1))
        alarms = np.sum(predicted & (y_true == 0))
        recall = caught / (caught + missed) if (caught + missed) else 0.0
        precision = caught / (caught + alarms) if (caught + alarms) else 0.0
        evaluated.append((float(threshold), float(recall), float(precision)))
    eligible = [item for item in evaluated if item[1] >= min_recall]
    pool = eligible or evaluated
    # Highest precision, then the higher threshold so ties raise fewer alarms.
    pool.sort(key=lambda item: (item[2], item[0]))
    return pool[-1][0]


def positive_scores(model, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def grouped_importance(model, x: pd.DataFrame, y: pd.Series, n_repeats: int = 20) -> list[dict]:
    baseline = average_precision_score(y, positive_scores(model, x))
    values = x.to_numpy()
    columns = list(x.columns)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, group in EXPLANATION_GROUPS:
        indexes = [columns.index(column) for column in group]
        drops = []
        for _ in range(n_repeats):
            shuffled = values.copy()
            shuffled[:, indexes] = values[rng.permutation(len(x))][:, indexes]
            frame = pd.DataFrame(shuffled, columns=columns)
            drops.append(baseline - average_precision_score(y, positive_scores(model, frame)))
        rows.append(
            {
                "feature": name,
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
            }
        )
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def preset_from_row(raw: pd.Series, preset_id: str, name: str, recorded: str, blurb: str) -> dict:
    return {
        "id": preset_id,
        "name": name,
        "blurb": blurb,
        "recorded": recorded,
        "product_id": str(raw["Product ID"]),
        "type": str(raw["Type"]),
        "air_temp": float(raw["Air temperature [K]"]),
        "process_temp": float(raw["Process temperature [K]"]),
        "rpm": int(raw["Rotational speed [rpm]"]),
        "torque": float(raw["Torque [Nm]"]),
        "tool_wear": int(raw["Tool wear [min]"]),
    }


def describe_failure(raw: pd.Series, mode: str) -> str:
    gap = float(raw["Process temperature [K]"] - raw["Air temperature [K]"])
    power_kw = float(raw["Torque [Nm]"] * raw["Rotational speed [rpm]"] * (2 * np.pi / 60) / 1000)
    wear_torque = float(raw["Torque [Nm]"] * raw["Tool wear [min]"])
    if mode == "Heat dissipation":
        return f"Recorded heat-dissipation failure. Gap {gap:.1f} K at {int(raw['Rotational speed [rpm]'])} rpm."
    if mode == "Power":
        return f"Recorded power failure. Mechanical power {power_kw:.1f} kW."
    if mode == "Overstrain":
        return (
            f"Recorded overstrain failure. Wear × torque {wear_torque:.0f} on a type {raw['Type']} part."
        )
    if mode == "Tool wear":
        return f"Recorded tool-wear failure. Tool wear {int(raw['Tool wear [min]'])} min."
    return "Recorded cycle with no failure."


def choose_presets(raw: pd.DataFrame, x: pd.DataFrame, y: pd.Series, modes: pd.Series, model) -> list[dict]:
    scores = pd.Series(positive_scores(model, x), index=x.index)
    sensor_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    healthy_mask = (y == 0) & (modes == "No failure")
    center = x.loc[y == 0, sensor_columns].median()
    scale = x.loc[y == 0, sensor_columns].std().replace(0, 1)
    distance = ((x.loc[healthy_mask, sensor_columns] - center) / scale).pow(2).sum(axis=1)
    healthy_index = distance.idxmin()
    specs = [
        ("healthy", "Healthy", healthy_index, "No failure"),
    ]
    for preset_id, mode in [
        ("heat", "Heat dissipation"),
        ("power", "Power"),
        ("overstrain", "Overstrain"),
        ("tool", "Tool wear"),
    ]:
        mask = (y == 1) & (modes == mode)
        if mask.sum() == 0:
            continue
        specs.append((preset_id, mode, scores.loc[mask].idxmax(), mode))

    presets = []
    for preset_id, name, index, recorded in specs:
        presets.append(
            preset_from_row(
                raw.loc[index],
                preset_id,
                name,
                recorded,
                describe_failure(raw.loc[index], recorded),
            )
        )
    return presets


def main() -> None:
    raw = load_raw()
    features = build_features(raw)
    leaked = FORBIDDEN_FEATURES & set(features.columns)
    if leaked:
        raise RuntimeError(f"Label columns leaked into the features: {sorted(leaked)}")

    y = raw["Machine failure"].astype(int)
    modes = failure_mode_label(raw)
    x_train, x_test, y_train, y_test, mode_train, mode_test, raw_train, raw_test = train_test_split(
        features,
        y,
        modes,
        raw,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    reports = []
    fitted = {}
    for name, estimator in candidates().items():
        oof = cross_val_predict(
            estimator,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = operating_threshold(y_train.to_numpy(), oof)
        model = clone(estimator).fit(x_train, y_train)
        holdout_scores = positive_scores(model, x_test)
        reports.append(
            {
                "name": name,
                "oof_average_precision": float(average_precision_score(y_train, oof)),
                "oof_roc_auc": float(roc_auc_score(y_train, oof)),
                "oof_at_threshold": scores_at(y_train.to_numpy(), oof, threshold),
                "holdout_average_precision": float(average_precision_score(y_test, holdout_scores)),
                "holdout_roc_auc": float(roc_auc_score(y_test, holdout_scores)),
                "holdout_at_threshold": scores_at(y_test.to_numpy(), holdout_scores, threshold),
            }
        )
        fitted[name] = (model, threshold)
        print(
            f"{name}: OOF AP {reports[-1]['oof_average_precision']:.3f} | "
            f"holdout recall {reports[-1]['holdout_at_threshold']['recall']:.3f} "
            f"precision {reports[-1]['holdout_at_threshold']['precision']:.3f}"
        )

    chosen_name = max(reports, key=lambda report: report["oof_average_precision"])["name"]
    model, threshold = fitted[chosen_name]
    print(f"Selected {chosen_name} by out-of-fold average precision.")

    mode_train_mask = mode_train.isin(PHYSICAL_MODES)
    mode_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    mode_model.fit(x_train.loc[mode_train_mask], mode_train.loc[mode_train_mask])
    mode_test_mask = mode_test.isin(PHYSICAL_MODES)
    mode_pred = mode_model.predict(x_test.loc[mode_test_mask])
    mode_true = mode_test.loc[mode_test_mask]
    mode_recall = {}
    for mode_name in PHYSICAL_MODES:
        mask = mode_true == mode_name
        mode_recall[mode_name] = float((mode_pred[mask.to_numpy()] == mode_name).mean()) if mask.any() else None

    eval_model = model
    holdout_scores = positive_scores(eval_model, x_test)
    review = pd.DataFrame(
        {
            "product_id": raw_test["Product ID"].astype(str).to_numpy(),
            "type": raw_test["Type"].astype(str).to_numpy(),
            "air_temp": raw_test["Air temperature [K]"].to_numpy(),
            "process_temp": raw_test["Process temperature [K]"].to_numpy(),
            "rpm": raw_test["Rotational speed [rpm]"].to_numpy(),
            "torque": raw_test["Torque [Nm]"].to_numpy(),
            "tool_wear": raw_test["Tool wear [min]"].to_numpy(),
            "recorded_failure": y_test.to_numpy().astype(int),
            "recorded_mode": mode_test.to_numpy(),
            "risk": holdout_scores,
            "inspect": (holdout_scores >= threshold).astype(int),
        }
    )
    review_path = ROOT / "data" / "holdout_review.csv"
    review.sort_values(["inspect", "risk"], ascending=[False, False]).to_csv(review_path, index=False)

    shift_parts = []
    shift_rng = np.random.default_rng(7)
    for mode_name in PHYSICAL_MODES:
        pool = review.index[review["recorded_mode"] == mode_name].to_numpy()
        if len(pool) == 0:
            continue
        shift_parts.append(review.loc[shift_rng.choice(pool, size=min(3, len(pool)), replace=False)])
    healthy_pool = review.index[review["recorded_failure"] == 0].to_numpy()
    shift_parts.append(review.loc[shift_rng.choice(healthy_pool, size=12, replace=False)])
    shift = pd.concat(shift_parts).sample(frac=1, random_state=7)
    shift_path = ROOT / "data" / "one_shift.csv"
    shift[["product_id", "type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]].to_csv(
        shift_path, index=False
    )

    # The holdout numbers above stay locked to eval_model. The copy the app
    # loads is fit again on every published row, at the same threshold.
    model = clone(candidates()[chosen_name]).fit(features, y)
    mode_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    mode_model.fit(features.loc[modes.isin(PHYSICAL_MODES)], modes.loc[modes.isin(PHYSICAL_MODES)])
    print(f"Refit {chosen_name} on all {len(raw):,} cycles.")

    baseline_raw = {
        "type": "M",
        "air_temp": float(raw["Air temperature [K]"].median()),
        "process_temp": float(raw["Process temperature [K]"].median()),
        "rpm": float(raw["Rotational speed [rpm]"].median()),
        "torque": float(raw["Torque [Nm]"].median()),
        "tool_wear": float(raw["Tool wear [min]"].median()),
    }
    baseline = build_features(pd.DataFrame([baseline_raw]))

    importance = grouped_importance(eval_model, x_test, y_test)
    presets = choose_presets(raw_test, x_test, y_test, mode_test, eval_model)
    flag_count = sum(raw[column].astype(int) for _, column in FAILURE_FLAGS)
    dataset = {
        "name": "AI4I 2020 Predictive Maintenance",
        "source": "UCI Machine Learning Repository",
        "rows": int(len(raw)),
        "failures": int(y.sum()),
        "failure_rate": float(y.mean()),
        "majority_class_accuracy": float(1 - y.mean()),
        "flag_counts": {column: int(raw[column].sum()) for _, column in FAILURE_FLAGS},
        "overlapping_failure_rows": int((flag_count > 1).sum()),
        "failure_without_flag": int(((y == 1) & (flag_count == 0)).sum()),
        "flag_without_failure": int(((y == 0) & (flag_count > 0)).sum()),
    }
    chosen_report = next(report for report in reports if report["name"] == chosen_name)
    metrics = {
        "dataset": dataset,
        "split": {
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "test_failures": int(y_test.sum()),
        },
        "excluded_inputs": ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"],
        "features": MODEL_FEATURES,
        "selection_rule": "Highest out-of-fold average precision on the training split.",
        "threshold_rule": (
            "On out-of-fold training scores, require at least 80% recall, "
            "then take the threshold with the highest precision."
        ),
        "chosen_model": chosen_name,
        "shipped_fit": {
            "rows": int(len(raw)),
            "note": (
                "Holdout metrics come from the model fit on the 8,000-row training split. "
                "After those numbers were locked, the same model was fit again on all 10,000 cycles. "
                "The app loads that second fit and keeps the threshold chosen on the training split."
            ),
        },
        "models": reports,
        "holdout": chosen_report["holdout_at_threshold"],
        "holdout_average_precision": chosen_report["holdout_average_precision"],
        "holdout_roc_auc": chosen_report["holdout_roc_auc"],
        "mode_recall": mode_recall,
        "importance": importance,
        "presets": presets,
        "baseline_risk": float(positive_scores(model, baseline)[0]),
    }

    for preset in presets:
        row = build_features(pd.DataFrame([preset]))
        risk = float(positive_scores(model, row)[0])
        preset["check_risk"] = risk
        print(f"Preset {preset['name']}: risk {risk:.3f} | {preset['blurb']}")

    healthy = next(preset["check_risk"] for preset in presets if preset["id"] == "healthy")
    for required in ("heat", "power", "overstrain"):
        risk = next(preset["check_risk"] for preset in presets if preset["id"] == required)
        if risk < threshold or healthy >= threshold:
            raise RuntimeError(
                f"Demo preset '{required}' does not separate from healthy "
                f"(healthy {healthy:.3f}, case {risk:.3f}, threshold {threshold:.3f})."
            )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "mode_model": mode_model,
        "mode_classes": list(mode_model.classes_),
        "threshold": float(threshold),
        "baseline_raw": baseline_raw,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
