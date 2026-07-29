import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------------------------
# 1. SETUP & LOAD ARTIFACTS
# -------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Load your saved models and column list from the local folder
    model = joblib.load("employee_attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("expected_columns.pkl")
    return model, scaler, expected_columns

model, scaler, EXPECTED_COLUMNS = load_artifacts()

# -------------------------------------------------------------------------
# 2. STREAMLIT UI - FRONTEND
# -------------------------------------------------------------------------
st.set_page_config(page_title="Attrition Predictor", page_icon="🏢")

st.title("Corporate Employee Attrition Predictor 🏢")
st.markdown("""
Enter the employee's details below to predict the likelihood of attrition. 
*Note: This form focuses on key HR features. Other variables (like Daily/Hourly rates) are set to their baseline corporate medians.*
""")

# Build the sidebar for user inputs 
st.sidebar.header("Key Employee Parameters")

monthly_income = st.sidebar.number_input("1. Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
age = st.sidebar.slider("2. Age", 18, 60, 36)
total_working_years = st.sidebar.slider("3. Total Working Years", 0, 40, 10)
monthly_rate = st.sidebar.number_input("4. Monthly Rate", min_value=2000, max_value=30000, value=14000)
distance_from_home = st.sidebar.slider("5. Distance From Home (miles)", 1, 30, 7)
years_at_company = st.sidebar.slider("6. Years At Company", 0, 40, 5)
overtime = st.sidebar.selectbox("7. OverTime", ["Yes", "No"])
num_companies_worked = st.sidebar.slider("8. Num Companies Worked", 0, 10, 2)

# -------------------------------------------------------------------------
# 3. PREDICTION LOGIC - BACKEND
# -------------------------------------------------------------------------
if st.button("Predict Attrition Risk", type="primary"):
    
    # Map 'Yes'/'No' back to 1/0 for your LabelEncoder
    overtime_encoded = 1 if overtime == "Yes" else 0
    
    # Create a base dictionary. 
    user_inputs = {
        # Active UI Inputs
        'MonthlyIncome': monthly_income,
        'Age': age,
        'TotalWorkingYears': total_working_years,
        'MonthlyRate': monthly_rate,
        'DistanceFromHome': distance_from_home,
        'YearsAtCompany': years_at_company,
        'OverTime': overtime_encoded,
        'NumCompaniesWorked': num_companies_worked,
        
        # Hidden Baseline Defaults (so the model doesn't crash)
        'DailyRate': 802,       # Median from dataset
        'HourlyRate': 66,       # Median from dataset
        'Education': 3, 
        'EnvironmentSatisfaction': 3, 
        'JobInvolvement': 3, 
        'JobLevel': 2, 
        'JobSatisfaction': 3, 
        'PerformanceRating': 3, 
        'StockOptionLevel': 1
    }

    # Convert the dictionary to a single-row DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Align the user's inputs to match the EXACT columns the model needs.
    input_df = input_df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    
    # Scale features using your saved scaler
    scaled_features = scaler.transform(input_df)
    
    # Make prediction using probabilities for better sensitivity
    probabilities = model.predict_proba(scaled_features)
    attrition_risk = probabilities[0][1] # Probability of "Yes" (Attrition)
    
    # Display the result
    st.divider()
    st.write(f"### Calculated Risk Probability: **{attrition_risk * 100:.1f}%**")
    st.progress(float(attrition_risk))
    
    # Using a 30% threshold due to the imbalanced dataset
    if attrition_risk >= 0.30:
        st.error("⚠️ **High Risk of Attrition.** This employee shows flight-risk patterns.")
    else:
        st.success("✅ **Low Risk of Attrition.** This employee is likely to stay.")