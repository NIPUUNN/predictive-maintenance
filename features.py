"""Feature engineering shared by training and the Streamlit app.

AI4I records five failure flags (TWF, HDF, PWF, OSF, RNF). Those are outcomes,
not sensors, so they are never model inputs. The extra columns below are the
physical quantities those failures depend on: temperature gap, mechanical
power, and wear times torque. The published numeric thresholds are not
hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

MODEL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff_k",
    "mechanical_power_w",
    "wear_torque",
    "type_L",
    "type_H",
]

EXPLANATION_GROUPS = [
    ("Air temperature", ["Air temperature [K]"]),
    ("Process temperature", ["Process temperature [K]"]),
    ("Rotational speed", ["Rotational speed [rpm]"]),
    ("Torque", ["Torque [Nm]"]),
    ("Tool wear", ["Tool wear [min]"]),
    ("Temperature gap", ["temp_diff_k"]),
    ("Mechanical power", ["mechanical_power_w"]),
    ("Wear × torque", ["wear_torque"]),
    ("Product type", ["type_L", "type_H"]),
]

SHORT_NAMES = {
    "type": "Type",
    "air_temp": "Air temperature [K]",
    "process_temp": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}

TYPE_MAP = {
    "L": "L",
    "M": "M",
    "H": "H",
    "l": "L",
    "m": "M",
    "h": "H",
    0: "L",
    1: "M",
    2: "H",
    "0": "L",
    "1": "M",
    "2": "H",
}

FAILURE_FLAGS = [
    ("Tool wear", "TWF"),
    ("Heat dissipation", "HDF"),
    ("Power", "PWF"),
    ("Overstrain", "OSF"),
    ("Random", "RNF"),
]

PHYSICAL_MODES = ["Tool wear", "Heat dissipation", "Power", "Overstrain"]


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Accept raw AI4I rows or a one-row form submission."""
    out = df.copy().rename(columns=SHORT_NAMES)
    missing = [column for column in SENSOR_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")
    out["Type"] = out["Type"].map(lambda value: TYPE_MAP.get(value, value))
    unknown = sorted(set(out["Type"]) - {"L", "M", "H"})
    if unknown:
        raise ValueError(f"Product type must be L, M, or H. Got {unknown}.")
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix. Product type M is the reference level."""
    src = normalize_frame(df)
    features = pd.DataFrame(index=src.index)
    features["Air temperature [K]"] = src["Air temperature [K]"].astype(float)
    features["Process temperature [K]"] = src["Process temperature [K]"].astype(float)
    features["Rotational speed [rpm]"] = src["Rotational speed [rpm]"].astype(float)
    features["Torque [Nm]"] = src["Torque [Nm]"].astype(float)
    features["Tool wear [min]"] = src["Tool wear [min]"].astype(float)
    features["temp_diff_k"] = (
        features["Process temperature [K]"] - features["Air temperature [K]"]
    )
    features["mechanical_power_w"] = (
        features["Torque [Nm]"] * features["Rotational speed [rpm]"] * (2 * np.pi / 60)
    )
    features["wear_torque"] = features["Torque [Nm]"] * features["Tool wear [min]"]
    product_type = src["Type"].astype(str)
    features["type_L"] = (product_type == "L").astype(float)
    features["type_H"] = (product_type == "H").astype(float)
    return features[MODEL_FEATURES]


def failure_mode_label(df: pd.DataFrame) -> pd.Series:
    """Single recorded mode. Overlapping flags become 'Multiple'."""
    count = sum(df[column].astype(int) for _, column in FAILURE_FLAGS)
    label = pd.Series("No failure", index=df.index, dtype=object)
    for name, column in FAILURE_FLAGS:
        label = label.mask((count == 1) & (df[column].astype(int) == 1), name)
    return label.mask(count > 1, "Multiple")


RAW_KNOBS = [
    ("Air temperature", "air_temp"),
    ("Process temperature", "process_temp"),
    ("Rotational speed", "rpm"),
    ("Torque", "torque"),
    ("Tool wear", "tool_wear"),
    ("Product type", "type"),
]


def local_contributions(predict_fn, current: dict, baseline: dict) -> pd.DataFrame:
    """How much each sensor moves risk relative to a typical machine.

    The sensor is reset to the training median and the engineered features are
    rebuilt. Positive means the live reading is pushing risk up.
    """

    def risk_of(payload: dict) -> float:
        return float(predict_fn(build_features(pd.DataFrame([payload])))[0])

    full = risk_of(current)
    records = []
    for name, key in RAW_KNOBS:
        swapped = dict(current)
        swapped[key] = baseline[key]
        records.append({"feature": name, "contribution": full - risk_of(swapped)})
    explained = pd.DataFrame(records)
    explained = explained.loc[explained["contribution"].abs() >= 0.005].copy()
    return explained.sort_values("contribution", ascending=False).reset_index(drop=True)
