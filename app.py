import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# 1. Page Configuration
st.set_page_config(
    page_title="ANN Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Enhanced 3D CSS Styling (Purple & Gold Theme)
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }

    .stApp {
        background: radial-gradient(circle at 50% 10%, #1D002B 0%, #0A0012 100%);
        color: #F3E8FF;
        font-family: 'Inter', sans-serif;
    }

    @keyframes float3D {
        0% { transform: translateY(0px) rotateX(0deg); }
        50% { transform: translateY(-8px) rotateX(3deg); }
        100% { transform: translateY(0px) rotateX(0deg); }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 10px 25px rgba(226, 183, 20, 0.2); }
        50% { box-shadow: 0 20px 45px rgba(226, 183, 20, 0.5); }
        100% { box-shadow: 0 10px 25px rgba(226, 183, 20, 0.2); }
    }

    .brand-hero-3d {
        background: linear-gradient(145deg, #2A0042, #180026);
        border: 1px solid rgba(226, 183, 20, 0.4);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8),
                    inset 0 2px 4px rgba(255, 255, 255, 0.15);
        animation: float3D 5s ease-in-out infinite;
    }

    .card-3d-container {
        background: linear-gradient(145deg, #1C002D, #11001C);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 8px 8px 20px #06000B, -6px -6px 20px #1A0029;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 15px;
    }

    .card-3d-container:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 12px 12px 25px #06000B, -8px -8px 25px #26003B;
        border-color: rgba(226, 183, 20, 0.4);
    }

    .card-title-bar {
        background: linear-gradient(90deg, #2A0042 0%, #3B005E 100%);
        border-bottom: 1px solid rgba(226, 183, 20, 0.3);
        padding: 12px 18px;
        margin: 0;
    }

    .card-title-bar h4 {
        color: #E2B714 !important;
        margin: 0 !important;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .card-body {
        padding: 18px;
    }

    .result-container-3d {
        background: linear-gradient(145deg, #25003A, #12001D);
        border: 2px solid #E2B714;
        padding: 35px;
        border-radius: 24px;
        text-align: center;
        animation: pulseGlow 2.5s infinite ease-in-out;
    }

    .stButton > button {
        background: linear-gradient(145deg, #FFD700, #B38F00);
        color: #0E021A !important;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: 1px;
        border-radius: 12px;
        height: 3.5em;
        width: 100%;
        border: none;
        box-shadow: 0 10px 25px rgba(226, 183, 20, 0.4);
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 15px 35px rgba(226, 183, 20, 0.6);
    }

    label {
        color: #E2B714 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model & Pipeline Loader
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('salary_model.keras')
    scaler = joblib.load('scaler.pkl')
    ohe = joblib.load('ohe.pkl')
    return model, scaler, ohe

try:
    model, scaler, ohe = load_assets()
except Exception as e:
    st.error(f"System Alert: Pipeline loading failed: {e}")
    st.stop()

# 4. Header UI
st.markdown("""
<div class="brand-hero-3d">
    <h1 style="margin: 0; color: #FFFFFF; font-size: 2.6rem; font-weight: 800;">
        ANN Salary Predictor
    </h1>
    <p style="margin-top: 8px; color: #E2B714; font-size: 1.05rem; font-weight: 600;">
        Deep Neural Network Compensation Engine
    </p>
</div>
""", unsafe_allow_html=True)

# 5. Input Matrix Form
st.markdown("<h3 style='color: #E2B714; margin-bottom: 20px;'>📋 Input Parameters</h3>", unsafe_allow_html=True)

with st.form("valuation_form_animated"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('''
        <div class="card-3d-container">
            <div class="card-title-bar">
                <h4>👤 Demographics</h4>
            </div>
            <div class="card-body">
        ''', unsafe_allow_html=True)
        geography = st.selectbox("Geography Region", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=38)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="card-3d-container">
            <div class="card-title-bar">
                <h4>💳 Credit & Wealth</h4>
            </div>
            <div class="card-body">
        ''', unsafe_allow_html=True)
        credit_score = st.slider("Credit Rating Score", min_value=300, max_value=850, value=650)
        balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=250000.0, value=75000.0, step=1000.0)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('''
        <div class="card-3d-container">
            <div class="card-title-bar">
                <h4>⚙️ Relationship & Churn</h4>
            </div>
            <div class="card-body">
        ''', unsafe_allow_html=True)
        num_of_products = st.selectbox("Active Products", [1, 2, 3, 4], index=0)
        has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active_member = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
        exited = st.radio("Customer Exited (Churn)?", ["Yes", "No"], horizontal=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    st.write("")
    submit_btn = st.form_submit_button("PREDICT SALARY")

# 6. Inference Execution
if submit_btn:
    with st.spinner("Processing feature tensor through Deep Neural Architecture..."):
        raw_input = pd.DataFrame([{
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active_member == "Yes" else 0,
            'Exited': 1 if exited == "Yes" else 0
        }])

        categorical_cols = ['Geography', 'Gender']
        numerical_cols = [c for c in raw_input.columns if c not in categorical_cols]

        encoded_array = ohe.transform(raw_input[categorical_cols])
        encoded_cols = ohe.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=raw_input.index)

        full_features = pd.concat([raw_input[numerical_cols], encoded_df], axis=1)
        scaled_input = scaler.transform(full_features)

        predicted_salary = float(model.predict(scaled_input, verbose=0)[0][0])

    st.markdown("---")

    st.markdown(f"""
    <div class="result-container-3d">
        <h4 style="color: #E2B714; margin: 0; font-size: 1.1rem; letter-spacing: 1.5px;">PREDICTED ESTIMATED SALARY</h4>
        <h1 style="color: #FFFFFF; font-size: 3.6rem; margin: 12px 0; font-weight: 800; text-shadow: 0 5px 15px rgba(0,0,0,0.8);">
            ${predicted_salary:,.2f}
        </h1>
        <p style="color: #00E676; font-size: 1rem; margin: 0; font-weight: 600;">✔ Neural Network Inference Complete</p>
    </div>
    """, unsafe_allow_html=True)
