import streamlit as st
import numpy as np
import pickle
from pathlib import Path
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.caption(
    "Reproducible ML system for multiclass student performance prediction"
)

MODELS_DIR = Path("artifacts/models")
METRICS_PATH = Path("artifacts/plots/metrics_summary.csv")

@st.cache_resource
def load_models():
    models = {}
    for model_file in MODELS_DIR.glob("*.pkl"):
        with open(model_file, "rb") as f:
            models[model_file.stem] = pickle.load(f)
    return models

models = load_models()

if not models:
    st.error("No trained models found. Please run the pipeline first.")
    st.stop()

st.sidebar.header("Model Selection")

model_names = list(models.keys())

default_model = "xgboost" if "xgboost" in model_names else model_names[0]

selected_model_name = st.sidebar.selectbox(
    "Choose model for prediction",
    options=model_names,
    index=model_names.index(default_model),
    help="XGBoost is recommended based on comparative evaluation"
)

selected_model = models[selected_model_name]


st.header("Student Input")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        score_1 = st.number_input(
            "Internal Score 1",
            min_value=0.0,
            max_value=20.0,
            value=10.0,
            step=0.5
        )
        score_2 = st.number_input(
            "Internal Score 2",
            min_value=0.0,
            max_value=20.0,
            value=10.0,
            step=0.5
        )

        study_time = st.selectbox(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            help="1 = very low, 4 = very high"
        )

    with col2:
        past_failures = st.number_input(
            "Past Failures",
            min_value=0,
            max_value=4,
            value=0,
            step=1
        )

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

    submit = st.form_submit_button("Predict Performance")

if submit:
    avg_internal_score = np.mean([score_1, score_2])
    score_variance = np.var([score_1, score_2])

    X_input = np.array([[
        avg_internal_score,
        score_variance,
        study_time,
        past_failures,
        absences
    ]])

    prediction = selected_model.predict(X_input)[0]

    st.subheader("Prediction Result")
    color_map = {
        "HIGH": "green",
        "MEDIUM": "orange",
        "LOW": "red"
    }

    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:#f5f5f5;">
            <h3>Predicted Performance:
            <span style="color:{color_map.get(prediction, 'black')};">
                {prediction}
            </span>
            </h3>
            <p><b>Model used:</b> {selected_model_name}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.header("📈 Model Comparison")

if METRICS_PATH.exists():
    metrics_df = pd.read_csv(METRICS_PATH)

    st.bar_chart(
        metrics_df.set_index("model")["accuracy"],
        use_container_width=True
    )

    st.dataframe(metrics_df, use_container_width=True)
else:
    st.warning("Metrics summary not found. Run the pipeline to generate metrics.")
