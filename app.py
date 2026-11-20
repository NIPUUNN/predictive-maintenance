"""Inspection desk for one milling cycle, or a shift of them.

The form uses the model fit on all 10,000 AI4I cycles. The test tab is a
locked 2,000-cycle holdout, scored before that final fit.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from features import FAILURE_FLAGS, build_features, failure_mode_label, local_contributions

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
REVIEW_PATH = ROOT / "data" / "holdout_review.csv"
SHIFT_PATH = ROOT / "data" / "one_shift.csv"

EXAMPLE_NAMES = {
    "healthy": "ordinary cycle",
    "heat": "tight gap, slow spindle",
    "power": "low torque",
    "overstrain": "worn tool, high torque",
    "tool": "long tool life",
}

READING_COLUMNS = {
    "type": "type",
    "product type": "type",
    "product_type": "type",
    "air_temp": "air_temp",
    "air temperature": "air_temp",
    "air temperature [k]": "air_temp",
    "process_temp": "process_temp",
    "process temperature": "process_temp",
    "process temperature [k]": "process_temp",
    "rpm": "rpm",
    "rotational speed": "rpm",
    "rotational speed [rpm]": "rpm",
    "torque": "torque",
    "torque [nm]": "torque",
    "tool_wear": "tool_wear",
    "tool wear": "tool_wear",
    "tool wear [min]": "tool_wear",
    "machine_id": "machine_id",
    "machine id": "machine_id",
    "product id": "machine_id",
    "product_id": "machine_id",
}

st.set_page_config(page_title="Milling inspection", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_review() -> pd.DataFrame:
    review = pd.read_csv(REVIEW_PATH)
    review["product_id"] = review["product_id"].astype(str)
    return review


@st.cache_data
def dataset_summary() -> dict:
    raw = pd.read_csv(DATA_PATH)
    modes = failure_mode_label(raw)
    preview_columns = [
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    ]
    return {
        "rows": int(len(raw)),
        "failures": int(raw["Machine failure"].sum()),
        "by_type": raw["Type"].value_counts().rename_axis("type").reset_index(name="cycles"),
        "by_mode": modes.value_counts().rename_axis("recorded").reset_index(name="cycles"),
        "preview": raw[preview_columns].head(12),
    }


@st.cache_data
def chart_frame() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    features = build_features(raw)
    modes = failure_mode_label(raw)
    plot = pd.DataFrame(
        {
            "power_kw": features["mechanical_power_w"] / 1000,
            "wear_torque": features["wear_torque"],
            "temp_gap_k": features["temp_diff_k"],
            "speed_rpm": features["Rotational speed [rpm]"],
            "outcome": modes,
        }
    )
    healthy = plot.loc[plot["outcome"] == "No failure"].sample(2000, random_state=42)
    failed = plot.loc[plot["outcome"] != "No failure"]
    return pd.concat([healthy, failed], ignore_index=True)


def risk_of(frame: pd.DataFrame) -> float:
    return float(bundle["model"].predict_proba(frame)[0, 1])


def score_table(readings: pd.DataFrame) -> pd.DataFrame:
    features = build_features(readings)
    scores = bundle["model"].predict_proba(features)[:, 1]
    modes = bundle["mode_model"].predict(features)
    cutoff = float(st.session_state.cutoff)
    scored = pd.DataFrame(
        {
            "risk": scores,
            "decision": ["Stop and inspect" if value >= cutoff else "Leave running" for value in scores],
            "similar_pattern": modes,
            "gap_k": features["temp_diff_k"].to_numpy(),
            "power_kw": features["mechanical_power_w"].to_numpy() / 1000,
            "wear_torque": features["wear_torque"].to_numpy(),
        },
        index=readings.index,
    )
    if "machine_id" in readings.columns:
        scored.insert(0, "machine", readings["machine_id"].astype(str).to_numpy())
    scored.insert(1 if "machine" in scored.columns else 0, "type", readings["type"].to_numpy())
    return scored


def coerce_readings(uploaded: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in uploaded.columns:
        spaced = " ".join(str(column).strip().lower().replace("_", " ").split())
        underscored = spaced.replace(" ", "_")
        if spaced in READING_COLUMNS:
            renamed[column] = READING_COLUMNS[spaced]
        elif underscored in READING_COLUMNS:
            renamed[column] = READING_COLUMNS[underscored]
    readings = uploaded.rename(columns=renamed)
    needed = ["type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]
    missing = [column for column in needed if column not in readings.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    return readings


def mark_edited() -> None:
    st.session_state.edited = True


def load_example() -> None:
    preset = next(item for item in presets if item["id"] == st.session_state.example)
    st.session_state.type = preset["type"]
    st.session_state.air = float(preset["air_temp"])
    st.session_state.process = float(preset["process_temp"])
    st.session_state.rpm = int(preset["rpm"])
    st.session_state.torque = float(preset["torque"])
    st.session_state.wear = int(preset["tool_wear"])
    st.session_state.machine_id = preset.get("product_id", "")
    st.session_state.edited = False


def current_payload() -> dict:
    return {
        "type": st.session_state.type,
        "air_temp": float(st.session_state.air),
        "process_temp": float(st.session_state.process),
        "rpm": float(st.session_state.rpm),
        "torque": float(st.session_state.torque),
        "tool_wear": float(st.session_state.wear),
    }


def inspection_reason(payload: dict, gap: float, power_kw: float, wear_torque: float, explained: pd.DataFrame, flagged: bool) -> str:
    if explained.empty or not flagged:
        return "Readings sit near a normal cycle. Leave the machine running."
    top = str(explained.iloc[0]["feature"])
    if top in {"Air temperature", "Process temperature", "Rotational speed"}:
        return (
            f"Temperature gap is {gap:.1f} K and the spindle is at {payload['rpm']:.0f} rpm. "
            f"{top} is the reading that moves the score."
        )
    if top == "Torque" and power_kw < 4.5:
        return f"Torque is {payload['torque']:.1f} N·m, so mechanical power is {power_kw:.1f} kW."
    if top in {"Torque", "Tool wear"} and wear_torque >= 9000:
        return (
            f"The tool has {payload['tool_wear']:.0f} min on it at {payload['torque']:.1f} N·m "
            f"(wear × torque {wear_torque:,.0f})."
        )
    if top == "Tool wear":
        return f"The tool has been cutting for {payload['tool_wear']:.0f} minutes."
    if top == "Torque":
        return f"Torque is {payload['torque']:.1f} N·m. Power for this cycle is {power_kw:.1f} kW."
    return f"{top} is the reading that moves the score."


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("Model file missing. Run python train.py in this folder.")
    st.stop()

metrics = bundle["metrics"]
presets = metrics["presets"]
dataset = metrics["dataset"]
if "example" not in st.session_state:
    st.session_state.example = presets[0]["id"]
    st.session_state.edited = False
    st.session_state.log = []
    st.session_state.cutoff = float(bundle["threshold"])
    load_example()

st.title("Milling inspection")
st.write(
    "A milling cycle finishes with six readings: product type, air temperature, "
    "process temperature, spindle speed, torque, and tool wear. "
    "The job of this page is to say which machines to stop before the next cycle, "
    "and which ones to leave running."
)
st.caption(
    f"Trained on {dataset['name']}, {dataset['source']}. "
    f"The published file has {dataset['rows']:,} cycles and {dataset['failures']:,} failures "
    f"({dataset['failure_rate']:.2%}). The form uses a model fit on all {dataset['rows']:,}. "
    "The test tab is 2,000 cycles scored before that final fit."
)

check_tab, test_tab, data_tab = st.tabs(
    ["Check a cycle", "Test on 2,000 held-out cycles", "The dataset"]
)

with check_tab:
    st.selectbox(
        "Example from the file",
        [preset["id"] for preset in presets],
        format_func=lambda preset_id: (
            f"{next(item['product_id'] for item in presets if item['id'] == preset_id)}"
            f" — {EXAMPLE_NAMES.get(preset_id, preset_id)}"
        ),
        key="example",
        on_change=load_example,
    )

    controls, result = st.columns([1, 1], gap="large")
    with controls:
        st.text_input("Machine", key="machine_id")
        st.selectbox("Product type", ["L", "M", "H"], key="type", on_change=mark_edited)
        st.slider("Air temperature (K)", 295.3, 304.5, step=0.1, key="air", on_change=mark_edited)
        st.slider("Process temperature (K)", 305.7, 313.8, step=0.1, key="process", on_change=mark_edited)
        st.slider("Rotational speed (rpm)", 1168, 2886, step=1, key="rpm", on_change=mark_edited)
        st.slider("Torque (N·m)", 3.8, 76.6, step=0.1, key="torque", on_change=mark_edited)
        st.slider("Tool wear (min)", 0, 253, step=1, key="wear", on_change=mark_edited)
        st.slider("Stop when risk is at least", 0.05, 0.95, step=0.01, key="cutoff")

    payload = current_payload()
    features = build_features(pd.DataFrame([payload]))
    risk = risk_of(features)
    gap = float(features["temp_diff_k"].iloc[0])
    power_kw = float(features["mechanical_power_w"].iloc[0] / 1000)
    wear_torque = float(features["wear_torque"].iloc[0])
    cutoff = float(st.session_state.cutoff)
    flagged = risk >= cutoff
    explained = local_contributions(
        lambda frame: bundle["model"].predict_proba(frame)[:, 1],
        payload,
        bundle["baseline_raw"],
    )
    active = None if st.session_state.edited else next(
        (preset for preset in presets if preset["id"] == st.session_state.example),
        None,
    )

    with result:
        st.metric("Failure risk", f"{risk:.1%}")
        st.write("Stop and inspect" if flagged else "Leave running")
        gap_col, power_col, wear_col = st.columns(3)
        gap_col.metric("Temp. gap", f"{gap:.1f} K")
        power_col.metric("Power", f"{power_kw:.1f} kW")
        wear_col.metric("Wear × torque", f"{wear_torque:,.0f}")
        st.write(inspection_reason(payload, gap, power_kw, wear_torque, explained, flagged))
        if flagged:
            mode_proba = bundle["mode_model"].predict_proba(features)[0]
            top_index = int(mode_proba.argmax())
            st.caption(
                f"If it is stopped, the closest recorded pattern is "
                f"{bundle['mode_classes'][top_index].lower()} "
                f"({mode_proba[top_index]:.0%} of trees)."
            )
        if active is not None:
            st.caption(
                f"In the file this cycle is labeled {active['recorded']}. "
                "It was included in the 10,000-row fit, so it is an example, not a test result."
            )
        if st.button("Add to this shift"):
            st.session_state.log.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "machine": st.session_state.get("machine_id") or "—",
                    "type": st.session_state.type,
                    "rpm": int(st.session_state.rpm),
                    "torque": round(float(st.session_state.torque), 1),
                    "wear_min": int(st.session_state.wear),
                    "risk": f"{risk:.1%}",
                    "decision": "Stop and inspect" if flagged else "Leave running",
                }
            )

    st.subheader("This shift")
    if st.session_state.log:
        log_frame = pd.DataFrame(st.session_state.log)
        st.dataframe(log_frame, hide_index=True, use_container_width=True)
        st.download_button(
            "Download this shift",
            log_frame.to_csv(index=False).encode("utf-8"),
            file_name="shift_log.csv",
            mime="text/csv",
        )
    else:
        st.caption("Nothing checked yet.")

    st.subheader("Score a list of cycles")
    st.caption(
        "Columns: type, air_temp, process_temp, rpm, torque, tool_wear. "
        "product_id is optional. This scores whatever you upload with the model fit on all 10,000 cycles."
    )
    if SHIFT_PATH.exists():
        st.download_button(
            "Download a 24-cycle example",
            SHIFT_PATH.read_bytes(),
            file_name="one_shift.csv",
            mime="text/csv",
        )
    upload = st.file_uploader("CSV of cycles", type="csv")
    if upload is not None:
        try:
            batch = score_table(coerce_readings(pd.read_csv(upload)))
        except ValueError as error:
            st.error(str(error))
        else:
            stops = int((batch["decision"] == "Stop and inspect").sum())
            st.write(f"{stops} of {len(batch)} would be stopped.")
            show = batch.sort_values("risk", ascending=False).copy()
            show["risk"] = show["risk"].map(lambda value: f"{value:.1%}")
            show["power_kw"] = show["power_kw"].map(lambda value: f"{value:.1f}")
            show["gap_k"] = show["gap_k"].map(lambda value: f"{value:.1f}")
            show["wear_torque"] = show["wear_torque"].map(lambda value: f"{value:,.0f}")
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.download_button(
                "Download scores",
                batch.to_csv(index=False).encode("utf-8"),
                file_name="cycle_scores.csv",
                mime="text/csv",
            )

with test_tab:
    holdout = metrics["holdout"]
    st.write(
        "These scores come from the random forest fit on 8,000 cycles. "
        "The 2,000 cycles here were not used to fit that model or to pick the threshold. "
        "After this test was saved, the app's model was fit again on all 10,000."
    )
    if not REVIEW_PATH.exists():
        st.error("data/holdout_review.csv is missing. Run python train.py.")
    else:
        review = load_review()
        recorded = review["recorded_failure"].astype(int)
        inspect = review["inspect"].astype(int)
        caught = int(((inspect == 1) & (recorded == 1)).sum())
        missed = int(((inspect == 0) & (recorded == 1)).sum())
        false_alarms = int(((inspect == 1) & (recorded == 0)).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cycles", f"{len(review):,}")
        c2.metric("Would stop", f"{int(inspect.sum()):,}")
        c3.metric("Real failures in that list", f"{caught} of {int(recorded.sum())}")
        c4.metric("Stopped by mistake", f"{false_alarms}")
        miss_modes = (
            review.loc[(inspect == 0) & (recorded == 1), "recorded_mode"]
            .value_counts()
        )
        miss_text = ", ".join(f"{int(count)} {name.lower()}" for name, count in miss_modes.items())
        st.caption(
            f"Missed failures: {missed} ({miss_text}). "
            f"Same holdout as the training report "
            f"({holdout['caught_failures']} caught, {holdout['false_alarms']} false alarms)."
        )
        view = st.radio(
            "Show",
            ["Inspection list", "Missed failures", "Unnecessary stops", "All 2,000"],
            horizontal=True,
        )
        shown = review
        if view == "Inspection list":
            shown = review.loc[inspect == 1]
        elif view == "Missed failures":
            shown = review.loc[(inspect == 0) & (recorded == 1)]
        elif view == "Unnecessary stops":
            shown = review.loc[(inspect == 1) & (recorded == 0)]
        table = shown.copy()
        table["risk"] = table["risk"].map(lambda value: f"{value:.1%}")
        table["decision"] = table["inspect"].map(lambda value: "Stop" if int(value) == 1 else "Run")
        table["recorded"] = table.apply(
            lambda row: row["recorded_mode"] if int(row["recorded_failure"]) == 1 else "No failure",
            axis=1,
        )
        table = table[
            ["product_id", "type", "rpm", "torque", "tool_wear", "risk", "decision", "recorded"]
        ].rename(
            columns={
                "product_id": "Machine",
                "type": "Type",
                "rpm": "rpm",
                "torque": "Torque",
                "tool_wear": "Tool wear",
                "risk": "Risk",
                "decision": "Decision",
                "recorded": "Recorded in file",
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)

with data_tab:
    summary = dataset_summary()
    st.write(
        f"{dataset['name']} is a synthetic milling process published by the "
        "[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). "
        f"The file in this project, data/ai4i2020.csv, is the whole set: "
        f"{summary['rows']:,} cycles, {summary['failures']:,} with Machine failure = 1."
    )
    st.write(
        "A model that never stops a machine is already right on "
        f"{dataset['majority_class_accuracy']:.1%} of cycles, so accuracy is a weak way to judge this. "
        "The useful question is how many real failures you catch per unnecessary stop."
    )
    left, right = st.columns(2)
    with left:
        st.caption("Cycles by product type")
        st.dataframe(summary["by_type"], hide_index=True, use_container_width=True)
    with right:
        st.caption("Recorded outcome")
        st.dataframe(summary["by_mode"], hide_index=True, use_container_width=True)
    st.caption("First rows. The failure flags in the raw file were not used as inputs.")
    st.dataframe(summary["preview"], hide_index=True, use_container_width=True)

    st.subheader("What was compared")
    st.write(metrics["shipped_fit"]["note"])
    rows = []
    for report in metrics["models"]:
        decision = report["holdout_at_threshold"]
        rows.append(
            {
                "Model": report["name"],
                "Used": "Yes" if report["name"] == metrics["chosen_model"] else "",
                "PR-AUC": f"{report['holdout_average_precision']:.3f}",
                "Recall": f"{decision['recall']:.1%}",
                "Precision": f"{decision['precision']:.1%}",
                "False alarms": decision["false_alarms"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Chosen by out-of-fold PR-AUC on the 8,000 training cycles. "
        "Inputs are the five sensors, product type, temperature gap, mechanical power, and wear × torque. "
        "Excluded: "
        + ", ".join(metrics["excluded_inputs"])
        + "."
    )

    points = chart_frame()
    color = alt.Color("outcome:N", title="Recorded outcome")
    power_chart = (
        alt.Chart(points)
        .mark_circle(opacity=0.7, size=36)
        .encode(
            x=alt.X("power_kw:Q", title="Mechanical power (kW)", scale=alt.Scale(domain=[0, 12])),
            y=alt.Y("wear_torque:Q", title="Wear × torque"),
            color=color,
            tooltip=["outcome", "power_kw", "wear_torque"],
        )
    )
    heat_chart = (
        alt.Chart(points)
        .mark_circle(opacity=0.7, size=36)
        .encode(
            x=alt.X("temp_gap_k:Q", title="Temperature gap (K)", scale=alt.Scale(domain=[7.4, 12])),
            y=alt.Y("speed_rpm:Q", title="Rotational speed (rpm)", scale=alt.Scale(domain=[1150, 2900])),
            color=color,
            tooltip=["outcome", "temp_gap_k", "speed_rpm"],
        )
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.altair_chart(power_chart, use_container_width=True)
    with chart_right:
        st.altair_chart(heat_chart, use_container_width=True)

    importance = pd.DataFrame(metrics["importance"])
    st.bar_chart(importance.set_index("feature")["importance"], horizontal=True)
    st.caption("Drop in holdout PR-AUC when that group of inputs is shuffled. Computed on the 2,000-cycle test.")
    flag_line = ", ".join(f"{name} {metrics['dataset']['flag_counts'][column]}" for name, column in FAILURE_FLAGS)
    st.caption(f"Recorded flags in the file, not model inputs: {flag_line}.")
