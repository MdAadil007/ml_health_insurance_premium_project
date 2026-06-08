import os
import streamlit as st
import pandas as pd
from joblib import load

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.stApp {
    background-color: #f5f7fa !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background-color: #f5f7fa !important;
}

[data-testid="stSidebar"] {
    background-color: #eef1f6 !important;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 960px;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.25rem;
    line-height: 1.2;
}

.main-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 2rem;
}

.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #9ca3af;
    margin-top: 2rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e5e7eb;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;

    color: #1f2937 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;

    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    color: #111827 !important;
}

div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

div[data-testid="stFormSubmitButton"] > button,
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}

div[data-testid="stFormSubmitButton"] > button:hover,
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.45) !important;
}

div[data-testid="stFormSubmitButton"] > button:active,
div.stButton > button:active {
    transform: translateY(0px) !important;
}

.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 16px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.badge-young {
    background-color: #dbeafe;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

.badge-rest {
    background-color: #ede9fe;
    color: #6d28d9;
    border: 1px solid #ddd6fe;
}

.result-card {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 20px;
    padding: 2.2rem 2rem;
    text-align: center;
    margin-top: 0.5rem;
    box-shadow: 0 16px 40px rgba(99,102,241,0.3);
}

.result-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.7);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

.result-amount {
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}

.result-sub {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
    margin-top: 0.4rem;
}

.error-card {
    background: #fef2f2;
    border: 1.5px solid #fecaca;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #991b1b;
    font-size: 0.88rem;
    margin-top: 0.5rem;
    line-height: 1.6;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# PATHS + MODEL LOADING
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "Artifacts")


@st.cache_resource
def load_artifacts():
    model_young = load(os.path.join(ARTIFACT_DIR, "best_model_young.joblib"))
    model_rest = load(os.path.join(ARTIFACT_DIR, "best_model_rest.joblib"))
    scaler_young = load(os.path.join(ARTIFACT_DIR, "scaler_young.joblib"))
    scaler_rest = load(os.path.join(ARTIFACT_DIR, "scaler_rest.joblib"))
    return model_young, model_rest, scaler_young, scaler_rest


try:
    model_young, model_rest, scaler_young_obj, scaler_rest_obj = load_artifacts()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ─────────────────────────────────────────
# MAPPINGS
# ─────────────────────────────────────────
MEDICAL_RISK_SCORES = {
    "No Disease": 0,
    "Thyroid": 5,
    "Diabetes": 6,
    "High blood pressure": 6,
    "Heart disease": 8,
    "Diabetes & Thyroid": 11,
    "Diabetes & High blood pressure": 12,
    "High blood pressure & Heart disease": 14,
    "Diabetes & Heart disease": 14,
}

INCOME_LEVEL_MAP = {
    "< 10L": 1,
    "10L - 25L": 2,
    "25L - 40L": 3,
    "> 40L": 4,
}

INSURANCE_PLAN_MAP = {
    "Bronze": 1,
    "Silver": 2,
    "Gold": 3,
}

AGE_SPLIT = 27

# Fallback order if the model does not expose feature names
FALLBACK_COLUMNS = [
    "age",
    "number_of_dependants",
    "income_level",
    "insurance_plan",
    "genetical_risk",
    "medical_risk_score",
    "gender_Male",
    "region_Northwest",
    "region_Southeast",
    "region_Southwest",
    "marital_status_Unmarried",
    "bmi_category_Obesity",
    "bmi_category_Overweight",
    "bmi_category_Underweight",
    "smoking_status_Occasional",
    "smoking_status_Regular",
    "employment_status_Salaried",
    "employment_status_Self-Employed",
]


def get_expected_columns(model):
    """Try to read the exact training feature order from the model."""
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        if cols:
            return cols

    if hasattr(model, "get_booster"):
        try:
            cols = model.get_booster().feature_names
            if cols:
                return cols
        except Exception:
            pass

    return FALLBACK_COLUMNS


def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = {
        # numeric / ordinal
        "age": inputs["age"],
        "number_of_dependants": inputs["number_of_dependants"],
        "income_level": INCOME_LEVEL_MAP[inputs["income_level"]],
        "insurance_plan": INSURANCE_PLAN_MAP[inputs["insurance_plan"]],
        "genetical_risk": inputs["genetical_risk"],
        "medical_risk_score": MEDICAL_RISK_SCORES[inputs["medical_history"]],

        # one-hot encodings
        "gender_Male": 1 if inputs["gender"] == "Male" else 0,
        "gender_Female": 1 if inputs["gender"] == "Female" else 0,

        "marital_status_Married": 1 if inputs["marital_status"] == "Married" else 0,
        "marital_status_Unmarried": 1 if inputs["marital_status"] == "Unmarried" else 0,

        "region_Northeast": 1 if inputs["region"] == "Northeast" else 0,
        "region_Northwest": 1 if inputs["region"] == "Northwest" else 0,
        "region_Southeast": 1 if inputs["region"] == "Southeast" else 0,
        "region_Southwest": 1 if inputs["region"] == "Southwest" else 0,

        "bmi_category_Normal": 1 if inputs["bmi_category"] == "Normal" else 0,
        "bmi_category_Obesity": 1 if inputs["bmi_category"] == "Obesity" else 0,
        "bmi_category_Overweight": 1 if inputs["bmi_category"] == "Overweight" else 0,
        "bmi_category_Underweight": 1 if inputs["bmi_category"] == "Underweight" else 0,

        "smoking_status_No Smoking": 1 if inputs["smoking_status"] == "No Smoking" else 0,
        "smoking_status_Occasional": 1 if inputs["smoking_status"] == "Occasional" else 0,
        "smoking_status_Regular": 1 if inputs["smoking_status"] == "Regular" else 0,

        "employment_status_Freelancer": 1 if inputs["employment_status"] == "Freelancer" else 0,
        "employment_status_Salaried": 1 if inputs["employment_status"] == "Salaried" else 0,
        "employment_status_Self-Employed": 1 if inputs["employment_status"] == "Self-Employed" else 0,
    }

    return pd.DataFrame([row])


def prepare_features(inputs: dict, model, scaler_data) -> pd.DataFrame:
    df = build_feature_row(inputs)

    expected_cols = get_expected_columns(model)
    df = df.reindex(columns=expected_cols, fill_value=0)
    df = df.astype(float)  # Fix: ensure all columns are float to avoid dtype conflict

    scaler = scaler_data["scaler"]
    scale_cols = [c for c in scaler_data["cols_to_scale"] if c in df.columns]

    if scale_cols:
        df.loc[:, scale_cols] = scaler.transform(df[scale_cols])

    return df


def predict_premium(inputs: dict):
    is_young = inputs["age"] <= AGE_SPLIT

    model = model_young if is_young else model_rest
    scaler_data = scaler_young_obj if is_young else scaler_rest_obj

    segment = f"Young Model (Age ≤ {AGE_SPLIT})" if is_young else f"Adult Model (Age > {AGE_SPLIT})"

    df = prepare_features(inputs, model, scaler_data)
    prediction = float(model.predict(df)[0])

    return round(prediction, 2), segment


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.markdown('<div class="main-title">🏥 Health Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Fill in your details below to estimate your annual health insurance premium.</div>',
    unsafe_allow_html=True,
)

if not models_loaded:
    st.error(
        f"⚠️ Could not load models. Make sure the <b>Artifacts</b> folder is inside the same folder as <b>main.py</b>.\n\n"
        f"Error: <code>{load_error}</code>",
        unsafe_allow_html=True,
    )
    st.stop()

with st.form("prediction_form"):
    st.markdown('<div class="section-header">Personal Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="input-label">Age</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=100, value=20, step=1, label_visibility="collapsed")
    with c2:
        st.markdown('<div class="input-label">Number of Dependants</div>', unsafe_allow_html=True)
        number_of_dependants = st.number_input(
            "Number of Dependants", min_value=0, max_value=20, value=0, step=1, label_visibility="collapsed"
        )
    with c3:
        st.markdown('<div class="input-label">Income Level</div>', unsafe_allow_html=True)
        income_level = st.selectbox("Income Level", list(INCOME_LEVEL_MAP.keys()), label_visibility="collapsed")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown('<div class="input-label">Genetical Risk (0–5)</div>', unsafe_allow_html=True)
        genetical_risk = st.number_input(
            "Genetical Risk", min_value=0, max_value=5, value=0, step=1, label_visibility="collapsed"
        )
    with c5:
        st.markdown('<div class="input-label">Insurance Plan</div>', unsafe_allow_html=True)
        insurance_plan = st.selectbox("Insurance Plan", list(INSURANCE_PLAN_MAP.keys()), label_visibility="collapsed")
    with c6:
        st.markdown('<div class="input-label">Employment Status</div>', unsafe_allow_html=True)
        employment_status = st.selectbox(
            "Employment Status", ["Freelancer", "Salaried", "Self-Employed"], label_visibility="collapsed"
        )

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        st.markdown('<div class="input-label">Gender</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"], label_visibility="collapsed")
    with c8:
        st.markdown('<div class="input-label">Marital Status</div>', unsafe_allow_html=True)
        marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"], label_visibility="collapsed")
    with c9:
        st.markdown('<div class="input-label">BMI Category</div>', unsafe_allow_html=True)
        bmi_category = st.selectbox("BMI Category", ["Normal", "Obesity", "Overweight", "Underweight"], label_visibility="collapsed")

    st.markdown('<div class="section-header">Health & Lifestyle</div>', unsafe_allow_html=True)
    c10, c11, c12 = st.columns(3)
    with c10:
        st.markdown('<div class="input-label">Smoking Status</div>', unsafe_allow_html=True)
        smoking_status = st.selectbox("Smoking Status", ["No Smoking", "Occasional", "Regular"], label_visibility="collapsed")
    with c11:
        st.markdown('<div class="input-label">Region</div>', unsafe_allow_html=True)
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"], label_visibility="collapsed")
    with c12:
        st.markdown('<div class="input-label">Medical History</div>', unsafe_allow_html=True)
        medical_history = st.selectbox("Medical History", list(MEDICAL_RISK_SCORES.keys()), label_visibility="collapsed")

    submitted = st.form_submit_button("🔍 Predict Premium")

if submitted:
    inputs = {
        "age": age,
        "number_of_dependants": number_of_dependants,
        "income_level": income_level,
        "insurance_plan": insurance_plan,
        "genetical_risk": genetical_risk,
        "employment_status": employment_status,
        "gender": gender,
        "marital_status": marital_status,
        "bmi_category": bmi_category,
        "smoking_status": smoking_status,
        "region": region,
        "medical_history": medical_history,
    }

    try:
        predicted, segment = predict_premium(inputs)

        st.markdown(
            f'<div class="model-badge {"badge-young" if "Young" in segment else "badge-rest"}">Model: {segment}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Annual Premium</div>
                <div class="result-amount">₹{predicted:,.0f}</div>
                <div class="result-sub">Estimated cost based on your profile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"Monthly estimate: ₹{predicted / 12:,.0f}")

    except Exception as e:
        st.markdown(
            f'<div class="error-card">⚠️ Prediction failed: {str(e)}</div>',
            unsafe_allow_html=True,
        )