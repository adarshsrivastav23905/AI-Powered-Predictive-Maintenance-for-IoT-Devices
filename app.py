from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_generator import generate_iot_dataset
from src.llm_assistant import OllamaError, generate_maintenance_advice, ollama_settings
from src.model_building import evaluate_model, train_model
from src.prediction import build_alert, predict_failure
from src.preprocessing import load_and_prepare_data


@st.cache_resource
def train_and_evaluate_project():
    data_path = Path("data/industrial_iot_failure_data.csv")
    if not data_path.exists():
        generate_iot_dataset(output_path=data_path, n_samples=3000, seed=42)

    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(data_path)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, model_dir="models")
    return data_path, X_test, y_test, feature_names, model, metrics


@st.cache_data
def load_dataset_preview():
    data_path = Path("data/industrial_iot_failure_data.csv")
    return pd.read_csv(data_path).head(20)


st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🛠️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #121a2b 40%, #1a2336 100%);
        color: #edf3ff;
    }
    .stSidebar {
        background: rgba(17, 24, 39, 0.92);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #162033;
        color: #dfeafc;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #8ad3ff;
    }
    div[data-testid="stMetricLabel"] {
        color: #dfeafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='padding: 0.8rem 0 1.2rem 0; display:flex; align-items:center; gap:14px;'>
      <img src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgdmlld0JveD0iMCAwIDQwMCA0MDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjQwMCIgaGVpZ2h0PSI0MDAiIGZpbGw9IiNCMTAyMDAiLz48dGV4dCB4PSI2MCIgeT0iMTQwIiBmaWxsPSIjRURGM0YxIiBmb250LWZhbWlseT0iQXJpYWwsIEhlbHZldGljYSwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSI0MSIgZm9udC13ZWlnaHQ9IjcwIj5BPC90ZXh0Pjx0ZXh0IHg9IjYwIiB5PSIxOTAiIGZpbGw9IiNBQ0Y2RkIiBmb250LWZhbWlseT0iQXJpYWwsIEhlbHZldGljYSwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSI0MSIgZm9udC13ZWlnaHQ9IjcwIj5JPC90ZXh0PjxwYXRoIGQ9Ik0xMzAgMjI1IEwxOTAgMjcwIDEwMDAgMTA1IDEyOTAgMjQxIiBzdHJva2U9IiM1N0Q1RkYiIHN0cm9rZS13aWR0aD0iOCIgZmlsbD0ibm9uZSIvPjxjaXJjbGUgY3g9IjE4MyIgY3k9IjE2MCIgcj0iMzAiIGZpbGw9IiM2RkVGQ0YiLz48L3N2Zz4=' />
      <div>
        <h1 style='margin:0; color:#EDF3FF;'>AI-Powered Predictive Maintenance Dashboard</h1>
        <p style='margin:4px 0 0 0; color:#A7C7FF;'>Industrial IoT health monitoring and failure-risk intelligence for proactive equipment maintenance</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    st.write("This dashboard evaluates a trained failure-prediction model using synthetic industrial sensor data.")
    regenerate = st.button("Generate fresh dataset and retrain model")
    if regenerate:
        data_path = Path("data/industrial_iot_failure_data.csv")
        generate_iot_dataset(output_path=data_path, n_samples=3000, seed=42)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Dataset regenerated and training cache cleared.")

    risk_threshold = st.slider("Risk threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    _, default_model = ollama_settings()
    ollama_model = st.text_input("Local Ollama model", value=default_model)
    st.caption(
        "Requests go to your configured Ollama endpoint (localhost by default); "
        "the app does not call a third-party LLM API."
    )

try:
    data_path, X_test, y_test, feature_names, model, metrics = train_and_evaluate_project()
except Exception as exc:  # pragma: no cover - UI fallback for startup issues
    st.error(f"Unable to train model: {exc}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col2.metric("Precision", f"{metrics['precision']:.3f}")
col3.metric("Recall", f"{metrics['recall']:.3f}")
col4.metric("F1 Score", f"{metrics['f1_score']:.3f}")

st.subheader("Operational Overview")
left, right = st.columns(2)
with left:
    st.write("Dataset path:", str(data_path))
    st.dataframe(load_dataset_preview(), use_container_width=True)

with right:
    risk_scores = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(risk_scores[y_test == 0], bins=20, alpha=0.7, label="Healthy", color="forestgreen")
    ax.hist(risk_scores[y_test == 1], bins=20, alpha=0.7, label="Failure", color="crimson")
    ax.axvline(risk_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={risk_threshold:.2f}")
    ax.set_title("Failure Probability Distribution")
    ax.set_xlabel("Predicted probability of failure")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

st.subheader("Feature Importance")
importances = model.named_steps["model"].feature_importances_
sorted_idx = importances.argsort()[::-1]
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.bar([feature_names[i] for i in sorted_idx[:10]], importances[sorted_idx[:10]], color="steelblue")
ax2.set_title("Top 10 Features Contributing to Failure Risk")
ax2.tick_params(axis="x", rotation=45)
st.pyplot(fig2)

st.subheader("Live Risk Prediction")
with st.form("risk_form"):
    sample = {
        "temperature": st.number_input("Temperature (°C)", value=98.0, min_value=20.0, max_value=150.0),
        "vibration": st.number_input("Vibration (mm/s)", value=3.8, min_value=0.1, max_value=10.0, step=0.1),
        "pressure": st.number_input("Pressure (psi)", value=155.0, min_value=30.0, max_value=200.0),
        "rpm": st.number_input("RPM", value=1765.0, min_value=500.0, max_value=2200.0),
        "current": st.number_input("Current (A)", value=62.0, min_value=5.0, max_value=90.0),
        "humidity": st.number_input("Humidity (%)", value=54.0, min_value=10.0, max_value=95.0),
        "coolant_temp": st.number_input("Coolant Temp (°C)", value=96.0, min_value=20.0, max_value=130.0),
        "power_draw": st.number_input("Power Draw (kW)", value=73.0, min_value=10.0, max_value=100.0),
        "device_age_hours": st.number_input("Device Age (hours)", value=4800.0, min_value=100.0, max_value=10000.0),
        "maintenance_gap_days": st.number_input("Maintenance Gap (days)", value=42.0, min_value=1.0, max_value=180.0),
    }
    submitted = st.form_submit_button("Predict failure risk")

if submitted:
    prediction, probability = predict_failure(model, sample)
    st.session_state["last_prediction"] = {
        "sample": sample,
        "prediction": prediction,
        "probability": probability,
    }

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    prediction = result["prediction"]
    probability = result["probability"]
    st.metric("Predicted class", "Failure likely" if prediction == 1 else "Machine healthy")
    st.progress(min(1.0, max(0.0, probability)))
    st.write(f"Probability of failure: {probability:.3f}")
    alert = build_alert(prediction, probability)
    st.warning(alert)

    st.subheader("Local LLM Maintenance Assistant")
    st.caption(
        "The LLM explains the classifier output and suggests inspection steps. "
        "It does not change the prediction or replace a technician."
    )
    if st.button("Generate maintenance explanation", type="primary"):
        try:
            with st.spinner(f"Requesting a local explanation from {ollama_model}..."):
                advice = generate_maintenance_advice(
                    result["sample"],
                    prediction,
                    probability,
                    alert,
                    model=ollama_model,
                )
            st.markdown(advice)
        except OllamaError as exc:
            st.error(str(exc))

st.markdown("---")
st.caption("Built for industrial monitoring, portfolio demonstration, and practical AI deployment storytelling.")
