import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Gauransh's Finance Analytics", layout="wide")
st.title("📊 Advanced Financial Health Dashboard")

# 2. Sidebar Inputs
st.sidebar.header("User Settings")
uploaded_file = st.sidebar.file_uploader("Upload Financial CSV", type="csv")
monthly_budget = st.sidebar.number_input("Monthly Budget Limit (₹)", min_value=0, value=50000)

if uploaded_file is not None:
    # --- Data Loading ---
    df = pd.read_csv(uploaded_file)
    
    # 3. FIX: Column Sanitization (Prevents KeyErrors)
    # Remove leading/trailing spaces and capitalize first letter
    df.columns = df.columns.str.strip().str.capitalize()

    # 4. Smart Mapping for your specific Dataset
    # This maps your CSV's complex names to simple names the app uses
    rename_dict = {
        'Monthly_expense_total': 'Amount',
        'Monthly_income': 'Income',
        'Savings_rate': 'Savings_rate',
        'Financial_stress_level': 'Stress',
        'Credit_score': 'Credit'
    }
    df = df.rename(columns=rename_dict)

    # 5. Fallback logic if 'Date' or 'Amount' are still missing
    if 'Date' not in df.columns:
        # Search for any column that has 'date' in the name
        cols_with_date = [c for c in df.columns if 'date' in c.lower()]
        if cols_with_date:
            df = df.rename(columns={cols_with_date[0]: 'Date'})
        else:
            st.error(f"❌ Could not find a 'Date' column. Columns found: {df.columns.tolist()}")
            st.stop()

    # Final conversion to DateTime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']) # Remove rows where date couldn't be parsed

    # 6. Dashboard Metrics
    total_spent = df['Amount'].sum() if 'Amount' in df.columns else 0
    remaining = monthly_budget - total_spent
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Expenses", f"₹{total_spent:,.2f}")
    m2.metric("Budget Goal", f"₹{monthly_budget:,.2f}")
    m3.metric("Remaining Balance", f"₹{remaining:,.2f}", delta=int(remaining))
    
    if 'Savings_rate' in df.columns:
        m4.metric("Avg Savings Rate", f"{df['Savings_rate'].mean():.1f}%")

    # 7. Visual Analytics Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Spending Analysis", "🧠 Financial Health", "📋 Raw Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Expenses by Category")
            if 'Category' in df.columns:
                st.bar_chart(df.groupby('Category')['Amount'].sum())
            else:
                st.info("No 'Category' column found for grouping.")
        with c2:
            st.subheader("Spending Trend")
            trend_df = df.set_index('Date').resample('D')['Amount'].sum()
            st.line_chart(trend_df)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Credit Score vs Stress Level")
            if 'Credit' in df.columns and 'Stress' in df.columns:
                fig, ax = plt.subplots()
                ax.scatter(df['Credit'], df['Stress'], alpha=0.5, color='purple')
                plt.xlabel("Credit Score")
                plt.ylabel("Stress Level")
                st.pyplot(fig)
        with c2:
            st.subheader("Budget Status")
            if total_spent > monthly_budget:
                st.error(f"You are over budget by ₹{abs(remaining):,.2f}")
            else:
                st.success("You are maintaining your budget well!")

    with tab3:
        st.dataframe(df, use_container_width=True)

    # 8. Data Export
    st.divider()
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📩 Export Analyzed Data", data=csv, file_name="gauransh_finance_report.csv")

else:
    st.info("👋 Welcome Gauransh! Please upload your financial dataset to begin.")
    # Show a hint of what's needed
    st.warning("Make sure your CSV has 'Date' and 'Monthly_expense_total' (or 'Amount') columns.")