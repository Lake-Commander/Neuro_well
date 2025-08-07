import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# === Page Configuration ===
st.set_page_config(page_title="Employee Burnout Predictor", layout="wide")

# === Load Model and Artifacts ===
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = {
        "Gender": joblib.load("models/Gender_encoder.pkl"),
        "Company Type": joblib.load("models/Company Type_encoder.pkl"),
        "WFH Setup Available": joblib.load("models/WFH Setup Available_encoder.pkl")
    }
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

# === Sidebar Navigation ===
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🤖 Predict Burnout", "📌 Recommendations"])

# === Dashboard Page ===
if page == "📊 Dashboard":
    st.title("📊 Burnout Insights Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Burn Rate Distribution")
        st.image("eda_outputs/burn_rate_distribution.png", use_container_width=True)

        st.subheader("Burn Rate by Company Type")
        st.image("eda_outputs/burnrate_by_company_type.png", use_container_width=True)

    with col2:
        st.subheader("Burn Rate vs Mental Fatigue")
        st.image("eda_outputs/burnrate_vs_fatigue.png", use_container_width=True)

        st.subheader("Burn Rate by Designation")
        st.image("eda_outputs/burnrate_by_designation.png", use_container_width=True)

    st.subheader("Correlation Heatmap")
    st.image("eda_outputs/correlation_heatmap.png", use_container_width=True)

    st.subheader("🔥 Feature Importance (Random Forest)")
    st.image("output_graphs/insights/feature_importance_plot.png", use_container_width=True)

# === Prediction Page ===
elif page == "🤖 Predict Burnout":
    st.title("🤖 Predict Employee Burnout")

    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        company_type = st.selectbox("Company Type", ["Service", "Product"])
        wfh_available = st.selectbox("WFH Setup Available", ["Yes", "No"])
        designation = st.slider("Designation (0 = Entry, 5 = Exec)", 0, 5, 2)
        resource_allocation = st.slider("Resource Allocation (1-10)", 1, 10, 5)
        mental_fatigue_score = st.slider("Mental Fatigue Score (0-10)", 0.0, 10.0, 5.0)
        tenure = st.slider("Tenure (Years)", 0, 40, 10)

        submitted = st.form_submit_button("Predict Burnout Risk")

    if submitted:
        # Encode categorical features
        gender_encoded = encoders["Gender"].transform([gender])[0]
        company_encoded = encoders["Company Type"].transform([company_type])[0]
        wfh_encoded = encoders["WFH Setup Available"].transform([wfh_available])[0]

        # Normalize numeric features
        numerical = scaler.transform([[resource_allocation, designation, tenure]])[0]

        # Assemble final input
        input_data = np.array([
            gender_encoded,
            company_encoded,
            wfh_encoded,
            numerical[0],  # Resource Allocation
            mental_fatigue_score,
            numerical[1],  # Designation
            numerical[2]   # Tenure
        ]).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)[0]
        prediction = np.clip(prediction, 0, 1)

        st.success(f"🧠 Estimated Burn Rate: **{prediction:.3f}**")
        if prediction > 0.7:
            st.error("⚠️ High burnout risk! Recommend immediate intervention.")
        elif prediction > 0.4:
            st.warning("🟠 Moderate burnout risk. Monitor closely.")
        else:
            st.info("🟢 Low burnout risk. Keep up the support!")

# === Recommendations Page ===
elif page == "📌 Recommendations":
    st.title("📌 Strategic Recommendations Based on Insights")

    st.markdown("""
### 🧠 Mental Fatigue Mitigation
- Mental Fatigue Score is the **top predictor** of burnout.
- ✅ **Action**: Launch wellness programs, encourage breaks, and offer counseling.

### 📈 Review Resource Allocation
- High workload is linked to higher burnout.
- ✅ **Action**: Monitor workloads using dashboards and automate reallocation if needed.

### 🏠 Enable Remote Work
- WFH flexibility shows correlation with lower burnout.
- ✅ **Action**: Offer flexible or hybrid setups with strong support infrastructure.

### 👥 Designation Sensitivity
- Junior staff are more burnout-prone.
- ✅ **Action**: Introduce mentorship, reduced KPIs, and learning tracks for juniors.

### 🧬 Gender & Company Type Patterns
- Burnout trends vary across groups.
- ✅ **Action**: Use inclusion data to craft fair, adaptive HR policies.
""")

    st.success("✅ These data-backed strategies can guide HR in reducing burnout and boosting employee wellbeing.")

# === Footer ===
st.markdown("---")
st.caption("Built with ❤️ by NeuroWell Analytics · For study purposes only · Not intended for clinical use.")
