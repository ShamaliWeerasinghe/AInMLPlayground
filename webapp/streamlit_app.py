from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Robot Predictive Maintenance", page_icon="🤖", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "ml_models" / "random_forest_failure_predictor.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train model first using ml/random_forest/train_and_deploy.py"
        )
    return joblib.load(MODEL_PATH)


def build_input_df(payload):
    return pd.DataFrame([payload])


def main():
    st.title("Industrial Robot Predictive Maintenance")
    st.caption("Interactive prediction UI powered by the trained RandomForest model.")

    try:
        model = load_model()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Sensor Inputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        timestamp = st.text_input("Timestamp", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        robot_id = st.selectbox("Robot ID", ["RB-101", "RB-102", "RB-103", "RB-104"])
        operating_hours = st.number_input("Operating Hours", min_value=0.0, value=320.5, step=0.1)
        temp_c = st.number_input("Temperature (C)", min_value=-20.0, value=81.4, step=0.1)

    with col2:
        vibration_mm_s = st.number_input("Vibration (mm/s)", min_value=0.0, value=5.2, step=0.1)
        motor_current_a = st.number_input("Motor Current (A)", min_value=0.0, value=20.3, step=0.1)
        hydraulic_pressure_bar = st.number_input("Hydraulic Pressure (bar)", min_value=0.0, value=101.1, step=0.1)
        ambient_humidity_pct = st.number_input("Ambient Humidity (%)", min_value=0.0, max_value=100.0, value=48.0, step=0.1)

    with col3:
        spindle_rpm = st.number_input("Spindle RPM", min_value=0.0, value=1498.0, step=0.1)
        input_voltage_v = st.number_input("Input Voltage (V)", min_value=0.0, value=400.9, step=0.1)
        error_code = st.selectbox("Error Code", ["E00", "E17", "E23", "E31"])
        maintenance_due_days = st.number_input("Maintenance Due (Days)", min_value=0, value=6, step=1)

    payload = {
        "timestamp": timestamp,
        "robot_id": robot_id,
        "operating_hours": operating_hours,
        "temp_c": temp_c,
        "vibration_mm_s": vibration_mm_s,
        "motor_current_a": motor_current_a,
        "hydraulic_pressure_bar": hydraulic_pressure_bar,
        "ambient_humidity_pct": ambient_humidity_pct,
        "spindle_rpm": spindle_rpm,
        "input_voltage_v": input_voltage_v,
        "error_code": error_code,
        "maintenance_due_days": maintenance_due_days,
    }

    if st.button("Predict Failure Risk", type="primary"):
        input_df = build_input_df(payload)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        st.subheader("Prediction Result")
        st.metric("Failure Probability (within 7 days)", f"{probability * 100:.2f}%")

        if prediction == 1:
            st.error("Likely failure within 7 days. Schedule maintenance immediately.")
        else:
            st.success("Unlikely failure within 7 days. Continue monitoring.")

        with st.expander("Input Payload"):
            st.json(payload)


if __name__ == "__main__":
    main()
