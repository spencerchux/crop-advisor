"""Farmer-facing Streamlit interface for Crop Advisor."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from app.api_client import CropAdvisorAPIError, CropAdvisorClient


st.set_page_config(
    page_title="Crop Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: #f6f8f3; }
        .block-container { max-width: 1080px; padding-top: 2rem; }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            color: white;
            background: linear-gradient(120deg, #174d36 0%, #2f7d4c 100%);
            margin-bottom: 1.25rem;
        }
        .hero h1 { margin: 0 0 0.25rem 0; font-size: 2.2rem; }
        .hero p { margin: 0; color: #e7f4e8; font-size: 1.05rem; }
        .recommendation {
            padding: 1.4rem;
            border: 1px solid #c9ddc8;
            border-left: 8px solid #2f7d4c;
            border-radius: 14px;
            background: white;
        }
        .recommendation-label { color: #56645b; font-size: 0.9rem; }
        .recommendation-crop {
            color: #174d36;
            font-size: 2rem;
            font-weight: 700;
            text-transform: capitalize;
        }
        div[data-testid="stForm"] {
            background: white;
            border: 1px solid #dde6da;
            border-radius: 16px;
            padding: 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client(base_url: str) -> CropAdvisorClient:
    return CropAdvisorClient(base_url=base_url)


def percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


default_api_url = os.getenv("CROP_ADVISOR_API_URL", "http://localhost:8000")

st.markdown(
    """
    <section class="hero">
        <h1>Crop Advisor</h1>
        <p>Use soil and weather measurements to find crops suited to your conditions.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input(
        "API URL",
        value=default_api_url,
        help="For local use, keep http://localhost:8000. On Render, use the deployed API URL.",
    )
    client = get_client(api_url)
    if st.button("Check API", use_container_width=True):
        try:
            health = client.health()
            if health.get("status") == "ok" and health.get("model_loaded"):
                st.success("API and model are ready")
            else:
                st.warning("API responded, but the model is not ready")
        except CropAdvisorAPIError as exc:
            st.error(str(exc))

    st.divider()
    st.caption(
        "This tool supports farm planning. Confirm important planting decisions with a local agronomist."
    )

st.subheader("Enter field measurements")
st.caption("Use values from a recent soil test and local weather records where possible.")

with st.form("crop_inputs"):
    soil_tab, weather_tab, context_tab = st.tabs(["Soil nutrients", "Weather", "Optional context"])

    with soil_tab:
        col1, col2, col3 = st.columns(3)
        nitrogen = col1.number_input(
            "Nitrogen (N)", min_value=0.0, max_value=500.0, value=70.0, step=1.0
        )
        phosphorus = col2.number_input(
            "Phosphorus (P)", min_value=0.0, max_value=500.0, value=40.0, step=1.0
        )
        potassium = col3.number_input(
            "Potassium (K)", min_value=0.0, max_value=500.0, value=45.0, step=1.0
        )
        ph = st.number_input(
            "Soil pH", min_value=0.1, max_value=14.0, value=5.54, step=0.01, format="%.2f"
        )

    with weather_tab:
        col1, col2 = st.columns(2)
        temperature = col1.number_input(
            "Temperature (°C)", min_value=-20.0, max_value=60.0, value=22.68, step=0.1
        )
        rainfall = col2.number_input(
            "Rainfall (dataset unit)",
            min_value=0.0,
            max_value=10_000.0,
            value=75.32,
            step=1.0,
            help="Use the same rainfall scale as the training dataset. Its source does not specify the unit.",
        )

    with context_tab:
        st.info("Humidity and location are collected for context but are not used by model version 1.")
        col1, col2 = st.columns(2)
        humidity_text = col1.text_input("Humidity (%)", placeholder="Example: 65")
        location = col2.text_input("Location", placeholder="Example: Kano, Nigeria")

    submitted = st.form_submit_button("Recommend a crop", type="primary", use_container_width=True)

if submitted:
    humidity = None
    if humidity_text.strip():
        try:
            humidity = float(humidity_text)
            if not 0 <= humidity <= 100:
                raise ValueError
        except ValueError:
            st.error("Humidity must be a number between 0 and 100.")
            st.stop()

    payload = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "ph": ph,
        "rainfall": rainfall,
        "temperature": temperature,
        "humidity": humidity,
        "location": location.strip() or None,
    }

    with st.spinner("Comparing your field conditions with the crop model..."):
        try:
            prediction = client.predict(payload)
        except CropAdvisorAPIError as exc:
            st.error(str(exc))
        else:
            crop = prediction["recommended_crop"]
            confidence = float(prediction["confidence"])
            st.markdown(
                f"""
                <section class="recommendation">
                    <div class="recommendation-label">Recommended crop</div>
                    <div class="recommendation-crop">{crop}</div>
                    <div>Model confidence: <strong>{percentage(confidence)}</strong></div>
                </section>
                """,
                unsafe_allow_html=True,
            )

            ranking = [{"Crop": crop.title(), "Probability": confidence}]
            ranking.extend(
                {
                    "Crop": item["crop"].title(),
                    "Probability": float(item["probability"]),
                }
                for item in prediction.get("alternatives", [])
            )
            ranking_frame = pd.DataFrame(ranking).set_index("Crop")

            st.subheader("Other likely crops")
            st.bar_chart(ranking_frame, y="Probability", color="#2f7d4c")
            display_frame = ranking_frame.copy()
            display_frame["Probability"] = display_frame["Probability"].map(percentage)
            st.dataframe(display_frame, use_container_width=True)

            for warning in prediction.get("warnings", []):
                st.warning(warning)

            st.caption(
                f"Prediction produced by {prediction.get('model_name', 'the trained model')}. "
                "Confidence is a model estimate, not a guarantee of crop performance."
            )

