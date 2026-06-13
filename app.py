import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS — cached so it is only read from disk once
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_css(file_name: str) -> str:
    with open(file_name, encoding="utf-8") as f:
        return f.read()

st.markdown(f"<style>{_load_css('style.css')}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Static HTML blocks — cached strings
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _hero_html() -> str:
    return """
<div class="hero-banner">
    <div class="heart-icon">❤️</div>
    <h1>Heart Disease Risk Predictor</h1>
    <p>Analyze key health indicators to assess potential heart disease risk.</p>
</div>"""

@st.cache_data(show_spinner=False)
def _ecg_html() -> str:
    return """
<div class="ecg-container">
  <svg class="ecg-svg" viewBox="0 0 700 60" preserveAspectRatio="none">
    <path class="ecg-path"
      d="M0,30 L60,30 L80,30 L90,10 L100,50 L110,5 L125,55 L135,30
         L200,30 L220,30 L230,10 L240,50 L250,5 L265,55 L275,30
         L340,30 L360,30 L370,10 L380,50 L390,5 L405,55 L415,30
         L480,30 L500,30 L510,10 L520,50 L530,5 L545,55 L555,30
         L620,30 L640,30 L650,10 L660,50 L670,5 L685,55 L700,30"/>
  </svg>
</div>"""

@st.cache_data(show_spinner=False)
def _info_cards_html() -> str:
    return """
<div class="patient-info-header">
    <span class="patient-info-icon">📋</span>
    <span class="patient-info-title">Patient Information</span>
</div>
<div class="info-cards-container">
    <div class="info-card">
        <div class="info-card-icon">📊</div>
        <div class="info-card-title">11 Health Parameters</div>
        <div class="info-card-desc">Analyze multiple cardiovascular indicators for accurate risk assessment.</div>
    </div>
    <div class="info-card">
        <div class="info-card-icon">🧠</div>
        <div class="info-card-title">AI-Powered Analysis</div>
        <div class="info-card-desc">Machine learning model trained on heart disease prediction data.</div>
    </div>
    <div class="info-card">
        <div class="info-card-icon">⚡</div>
        <div class="info-card-title">Instant Results</div>
        <div class="info-card-desc">Generate risk predictions within seconds after submission.</div>
    </div>
    <div class="info-card">
        <div class="info-card-icon">🔒</div>
        <div class="info-card-title">Private &amp; Secure</div>
        <div class="info-card-desc">Patient information is processed securely and not stored.</div>
    </div>
</div>"""

# ─────────────────────────────────────────────
# Hero header + ECG (static — rendered once)
# ─────────────────────────────────────────────
st.markdown(_hero_html(), unsafe_allow_html=True)
st.markdown(_ecg_html(),  unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model artefacts — cached across reruns
# ─────────────────────────────────────────────
MODEL_FILES = {
    "model":   "SVM2_Spam.pkl",
    "scaler":  "scaler2.pkl",
    "columns": "columns2.pkl",
}

@st.cache_resource(show_spinner=False)
def load_artefacts():
    loaded, missing = {}, []
    for key, fname in MODEL_FILES.items():
        if os.path.exists(fname):
            loaded[key] = joblib.load(fname)
        else:
            missing.append(fname)
    return loaded, missing

artefacts, missing_files = load_artefacts()

if missing_files:
    st.error(
        f"⚠️ Missing model files: {', '.join(missing_files)}\n\n"
        "Place **SVM2_Spam.pkl**, **scaler2.pkl**, and **columns2.pkl** "
        "in the same folder as app.py and restart."
    )
    st.stop()

model         = artefacts["model"]
scaler        = artefacts["scaler"]
expected_cols = artefacts["columns"]

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────
SPACER = "<div style='height:22px'></div>"

form_container = st.container(border=True)
with form_container:
    st.markdown(_info_cards_html(), unsafe_allow_html=True)

    # Row 1 — numeric sliders
    r1c1, r1c2, r1c3 = st.columns(3, gap="large")
    with r1c1:
        age         = st.slider("Age", 18, 100, 45)
    with r1c2:
        resting_bp  = st.slider("Resting BP (mm Hg)", 80, 200, 120)
    with r1c3:
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)

    st.markdown(SPACER, unsafe_allow_html=True)

    # Row 2 — numeric sliders + selectbox
    r2c1, r2c2, r2c3 = st.columns(3, gap="large")
    with r2c1:
        max_hr    = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    with r2c2:
        oldpeak   = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 0.0, step=0.1)
    with r2c3:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", [0, 1],
            format_func=lambda x: "Yes (>120)" if x == 1 else "No (≤120)"
        )

    st.markdown(SPACER, unsafe_allow_html=True)

    # Row 3 — categorical selectboxes
    r3c1, r3c2, r3c3 = st.columns(3, gap="large")
    with r3c1:
        gender     = st.selectbox("Gender", ["Male", "Female"])
    with r3c2:
        chest_pain = st.selectbox(
            "Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
            help="ATA=Atypical Angina, NAP=Non-Anginal, ASY=Asymptomatic, TA=Typical Angina"
        )
    with r3c3:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

    st.markdown(SPACER, unsafe_allow_html=True)

    # Row 4 — categorical selectboxes
    r4c1, r4c2, _ = st.columns(3, gap="large")
    with r4c1:
        exercise_ang = st.selectbox("Exercise Induced Angina", ["N", "Y"])
    with r4c2:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ── Stat badges — built once per rerun (values change with sliders)
st.markdown(
    f'<span class="stat-badge">Age: {age}</span>'
    f'<span class="stat-badge">BP: {resting_bp}</span>'
    f'<span class="stat-badge">Chol: {cholesterol}</span>'
    f'<span class="stat-badge">MaxHR: {max_hr}</span>'
    f'<span class="stat-badge">Oldpeak: {oldpeak}</span>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Predict button
# ─────────────────────────────────────────────
predict_clicked = st.button("🔍  Analyze Cardiovascular Risk")

if predict_clicked:

    # 1. Build input row
    raw = {
        "Age":            age,
        "RestingBP":      resting_bp,
        "Cholesterol":    cholesterol,
        "FastingBS":      int(fasting_bs),
        "MaxHR":          max_hr,
        "Oldpeak":        float(oldpeak),
        "Sex":            gender,
        "ChestPainType":  chest_pain,
        "RestingECG":     resting_ecg,
        "ExerciseAngina": exercise_ang,
        "ST_Slope":       st_slope,
    }

    # 2. One-hot encode
    input_df      = pd.DataFrame([raw])
    cat_cols      = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    input_encoded = pd.get_dummies(input_df, columns=cat_cols)

    # 3. Align to training schema
    input_aligned = input_encoded.reindex(columns=expected_cols, fill_value=0).astype(float)

    # 4. Scale + predict
    input_scaled = scaler.transform(input_aligned)
    prediction   = model.predict(input_scaled)[0]

    # 5. Show result card
    if prediction == 1:
        st.markdown("""
        <div class="result-card danger">
            <div class="result-icon">🚨</div>
            <h1>Elevated Risk of Heart Disease Detected</h1>
            <h4>The entered health parameters indicate potential cardiovascular risk factors.
            This result is an AI-based assessment and should not be considered a medical diagnosis.
            Consider consulting a healthcare professional for further evaluation and guidance.</h4>
            <h3>❤️ Consult a Healthcare Professional</h3>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card safe">
            <div class="result-icon">💚</div>
            <h1>Low Risk — Healthy Heart Indicators</h1>
            <h4>Based on the provided health parameters, your cardiovascular risk appears to be low.
            Your current measurements are within a generally healthy range. Continue maintaining
            a balanced diet, regular physical activity, and routine health check-ups.</h4>
            <h3>💚 Maintain a Healthy Lifestyle</h3>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#4a5568;font-size:0.8rem;margin-top:3rem;padding-bottom:1rem;">'
    'Heart Disease Risk Predictor · SVM Model · For research &amp; educational use only.<br>'
    'Not a substitute for professional medical advice.</div>',
    unsafe_allow_html=True,
)
