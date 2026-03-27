import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

# --- Load Model & Data ---
@st.cache_resource
def load_resources():
    # Gets the absolute path to the folder where app.py is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_path, 'models', 'best_model.pkl')
    data_path = os.path.join(base_path, 'data', 'X_test.csv')
    
    # Safety Check: Show a nice error message if files are still missing
    if not os.path.exists(data_path):
        st.error(f"⚠️ Data file not found on server at: {data_path}")
        st.stop()
        
    model = joblib.load(model_path)
    X_test = pd.read_csv(data_path)
    return model, X_test

# --- CRITICAL FIX: CALL THE FUNCTION HERE ---
# This makes 'model' and 'X_test' available to the rest of the script
model, X_test = load_resources()

# --- Helper: Get Feature Importance ---
def get_importance_df(model):
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Extract names from OneHotEncoder and Numerical Scaler
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out()
    num_features = preprocessor.transformers_[0][2]
    all_feature_names = list(num_features) + list(cat_features)
    
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': classifier.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)
    
    return importance_df

# --- SIDEBAR: Control Panel ---
st.sidebar.header("🕹️ Employee Simulator")
st.sidebar.write("Modify parameters to see real-time predictions.")

# Now 'X_test' is defined and this line will work!
sample_idx = st.sidebar.number_input("Select Employee Index", 0, len(X_test)-1, 0)
input_data = X_test.iloc[[sample_idx]].copy()

st.sidebar.divider()
st.sidebar.subheader("Adjust Factors")

overtime_input = st.sidebar.selectbox(
    "Overtime Status", 
    ["Yes", "No"], 
    index=0 if input_data['OverTime'].values[0] == 'Yes' else 1
)

income_input = st.sidebar.slider(
    "Monthly Income ($)", 
    1000, 20000, 
    int(input_data['MonthlyIncome'].values[0])
)

st.sidebar.info("💡 Changes made here will update the 'Risk Probability' in the main window immediately.")

# Update the sample data with sidebar inputs
input_data['OverTime'] = overtime_input
input_data['MonthlyIncome'] = income_input

# --- MAIN LAYOUT ---
st.title("📊 HR Intelligence: Employee Attrition Portal")
st.markdown("This dashboard leverages **Random Forest + SMOTE** to analyze and predict employee turnover.")

tab1, tab2 = st.tabs(["🎯 Live Prediction", "📈 Model Insights"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Attrition Risk")
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        # Color coding based on risk
        color = "#FF4B4B" if prediction_proba > 0.5 else "#28a745"
        
        # Display Probability
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{prediction_proba*100:.1f}%</h1>", unsafe_allow_html=True)
        st.progress(prediction_proba) # Visual progress bar
        
        if prediction_proba > 0.5:
            st.error("⚠️ **HIGH RISK**: This employee shows patterns common in those who leave.")
        else:
            st.success("✅ **LOW RISK**: This employee is likely to stay.")

    with col2:
        st.subheader("Current Profile Snapshot")
        display_cols = ['Age', 'JobRole', 'Department', 'YearsAtCompany', 'MonthlyIncome', 'OverTime']
        st.table(input_data[display_cols])

with tab2:
    st.header("Why are employees leaving?")
    
    col_img, col_chart = st.columns(2)
    
    with col_img:
        st.subheader("Confusion Matrix")
        # Added absolute path check for image as well
        base_path = os.path.dirname(os.path.abspath(__file__))
        cm_path = os.path.join(base_path, 'models', 'confusion_matrix.png')
        
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.warning("Confusion Matrix image not found in 'models/'. Run evaluate.py first.")
    
    with col_chart:
        st.subheader("Top 10 Global Drivers")
        importance_df = get_importance_df(model)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma', ax=ax)
        ax.set_title("Impact Score on Model Decision")
        st.pyplot(fig)

st.divider()
st.caption("3rd Year Project - Developed by Gauransh | Built with Python, Scikit-Learn, and Streamlit")
