import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from streamlit_lottie import st_lottie
import requests


df = pd.read_csv(r"C:\Users\Vignesh\OneDrive\Desktop\Datascience\PROJECT1\Employee\Employee-Attrition_Processed.csv")

st.set_page_config(page_title="Employee Attrition Analysis", page_icon="👥", layout="wide",initial_sidebar_state="collapsed")

page = st.sidebar.selectbox("📂 Navigate", ["📋 Employee Records", "📊 Analytics & Prediction"])

st.markdown("""
    <style>
        html, body, [class*="st-"], .stApp {
            color: darkblue!important;
        }

    [data-testid="stAppViewContainer"] {
        background-image: url("https://i.ibb.co/v6L9W8X2/Chat-GPT-Image-Aug-16-2025-10-35-57-AM.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }
    
    </style>
""", unsafe_allow_html=True)

st.title("🧑‍💼 Employee Attrition Analysis and Prediction")

if page == "📋 Employee Records":
    
    st.sidebar.header("🔍 Filters")

    department = st.sidebar.selectbox(
        "Department",
        ["All", "Human Resources", "Research & Development", "Sales"]
    )

    gender = st.sidebar.multiselect(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.sidebar.selectbox(
        "Marital Status",
        ["All", "Single", "Married", "Divorced"]
    )

    job_level = st.sidebar.selectbox(
        "Job Level",
        ["All", "Entry_Level", "Junior_Level", "Mid_Level", "Senior_Level", "Executive_Level"]
    )

    age = st.sidebar.slider("Age Range", 18, 60, (18, 60))


    show_attrition = st.sidebar.checkbox("Filter: Attrition Only")

    
    st.success(f"Total records : {len(df)}")
    st.header("🗃️ View & Filter Records")

    filtered_df = df.copy()

    if department != "All":
        filtered_df = filtered_df[filtered_df["Department"] == department]

    if gender:
        filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

    if marital_status != "All":
        filtered_df = filtered_df[filtered_df["MaritalStatus"] == marital_status]

    if job_level != "All":
        filtered_df = filtered_df[filtered_df["JobLevel"] == job_level]

    filtered_df = filtered_df[(filtered_df["Age"] >= age[0]) & (filtered_df["Age"] <= age[1])]

    if show_attrition:
        filtered_df = filtered_df[filtered_df["Attrition"] == "Yes"]

    st.dataframe(filtered_df, use_container_width=True)
    st.warning(f"Filtered Records: {len(filtered_df)}")

    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data", data=csv, file_name="filtered_data.csv", mime="text/csv")


    st.subheader("📊 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Employees", len(filtered_df))
    with col2:
        st.metric("Attrition Rate", f"{(filtered_df['Attrition'].map({'Yes': 1, 'No': 0}).mean() * 100):.2f}%")
    with col3:
        st.metric("Average Age", f"{filtered_df['Age'].mean():.1f} years")
    with col4:
        st.metric("Most Common Job Level", filtered_df['JobLevel'].mode().iloc[0] if not filtered_df['JobLevel'].mode().empty else "N/A")

elif page == "📊 Analytics & Prediction":
    st.sidebar.header("Analysis")
    analysis_section = st.sidebar.selectbox("📊 Select Analysis Type",["📊 Employee Insights","🤖 Predictive Models"])

    if analysis_section == "📊 Employee Insights":
        st.subheader("📈 Employee Insights Dashboard")
        st.markdown("Visualize key employee metrics and trends.")

        def load_lottieurl(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

        lottie_animation = load_lottieurl("https://lottie.host/66ed66e2-ef49-4bad-b40e-46b5c6167b4b/UchFEkcbET.json")

        with st.sidebar:
            st_lottie(lottie_animation, speed=1, width=250, height=300, key="welcome_anim")

        df = pd.read_csv(r"C:\Users\Vignesh\OneDrive\Desktop\Datascience\PROJECT1\Employee\Employee-Attrition_Processed.csv")

     
        df['Attrition_Flag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

        
        df['JobSatisfaction_Label'] = df['JobSatisfaction'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4})
        df['PerformanceRating_Label'] = df['PerformanceRating'].map({'Excellent': 3, 'Outstanding': 4})
        df['WorkLifeBalance_Label'] = df['WorkLifeBalance'].map({'Bad': 1, 'Good': 2, 'Better': 3, 'Best': 4})

        # KPI Metrics 
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            attrition_rate = df['Attrition_Flag'].mean() * 100
            st.metric("Attrition Rate", f"{attrition_rate:.2f}%")

        with kpi2:
            avg_job_sat = df['JobSatisfaction_Label'].mean()
            st.metric("Avg Job Satisfaction", f"{avg_job_sat:.2f}")

        with kpi3:
            avg_perf = df['PerformanceRating_Label'].mean()
            st.metric("Avg Performance Rating", f"{avg_perf:.2f}")

        with kpi4:
            avg_wlb = df['WorkLifeBalance_Label'].mean()
            st.metric("Avg Work-Life Balance", f"{avg_wlb:.2f}")

        st.markdown("---")

        df['JobSatisfaction_Label'] = pd.to_numeric(df['JobSatisfaction_Label'], errors='coerce')
        df['WorkLifeBalance_Label'] = pd.to_numeric(df['WorkLifeBalance_Label'], errors='coerce')
        df['PerformanceRating_Label'] = pd.to_numeric(df['PerformanceRating_Label'], errors='coerce')

        df['Risk_Score'] = (
            (df['JobSatisfaction_Label'] <= 2).astype(int) * 2 +
            (df['OverTime'] == 'Yes').astype(int) * 2 +
            (df['WorkLifeBalance_Label'] <= 2).astype(int) * 2 +
            (df['PerformanceRating_Label'] <= 3).astype(int)
        )

        df['High_Risk'] = (df['Risk_Score'] >= 4).astype(int)
        
        # 🚨 Summary Metrics
        total_employees = len(df)

        # High-Risk
        high_risk_count = df['High_Risk'].sum()
        high_risk_percent = (high_risk_count / total_employees) * 100

        # High Job Satisfaction (JobSatisfaction_Label >= 4)
        high_satisfaction_count = (df['JobSatisfaction_Label'] >= 4).sum()
        high_satisfaction_percent = (high_satisfaction_count / total_employees) * 100

        # High Performance (PerformanceRating_Label >= 4)
        high_perf_count = (df['PerformanceRating_Label'] >= 4).sum()
        high_perf_percent = (high_perf_count / total_employees) * 100

        st.subheader("📊 Employee Risk & Engagement Metrics")

        # Show summary metrics in two rows
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Employees", total_employees)
        col2.metric("High-Risk Employees", high_risk_count, f"{high_risk_percent:.1f}%")
        col3.metric("High Job Satisfaction", high_satisfaction_count, f"{high_satisfaction_percent:.1f}%")

        col4, col5, _ = st.columns(3)
        col4.metric("High Performance Employees", high_perf_count, f"{high_perf_percent:.1f}%")


        st.subheader("🚨 High-Risk Employees(Top 10)")
        high_risk = df[df['High_Risk'] == 1][[
            'Age','Department','JobRole',
            'JobSatisfaction_Label','WorkLifeBalance_Label',
            'OverTime','PerformanceRating_Label'
        ]].head(10)

        st.dataframe(high_risk, use_container_width=True)



        st.subheader("😊 High Job Satisfaction(Top 10)")
        top_sat = df[['Age','Department','JobRole','JobSatisfaction','Attrition']].sort_values(by='JobSatisfaction', ascending=False).head(10)
        st.dataframe(top_sat, use_container_width=True)

        st.subheader("🏆 High Performance Score(Top 10)")
        top_perf = df[['Age','Department','JobRole','PerformanceRating','Attrition']].sort_values(by='PerformanceRating', ascending=False).head(10)
        st.dataframe(top_perf, use_container_width=True)



    elif analysis_section == "🤖 Predictive Models":
        st.markdown("### Model Training and Evaluation")
        st.markdown("We will use Random Forest Classifier to predict employee attrition based on various features.")


        attrition_model = joblib.load("C:\\Users\\Vignesh\\Data science\\python\\Employee\\attrition_model.pkl")
        performance_model = joblib.load("C:\\Users\\Vignesh\\Data science\\python\\Employee\\PerformanceRating_model.pkl")
        promotion_model = joblib.load("C:\\Users\\Vignesh\\Data science\\python\\Employee\\Promotion_model.pkl")

        st.sidebar.header("Navigation")
        choice = st.sidebar.radio(
            "Select Prediction Task",
            ["Attrition Risk", "Performance Prediction", "Promotion Prediction"]
        )

        def load_lottieurl(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

        lottie_animation = load_lottieurl("https://lottie.host/919b5dd5-b732-4c29-99d7-3d0380f3c908/H4Wn6k1OMl.json")

        with st.sidebar:
            st_lottie(lottie_animation, speed=1, width=250, height=300, key="welcome_anim")

        # -------------------------
        if choice == "Attrition Risk":
            st.subheader("⚠️ Attrition Risk Prediction")

            gender = st.selectbox("Gender", ["Male", "Female"])
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            jobrole = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician", "Manager",
                "Manufacturing Director", "Healthcare Representative", "Research Director",
                "Human Resources", "Other"
            ])
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            jobsatisfaction = st.selectbox("Job Satisfaction", ["Low","Medium","High","Very High"])
            age = st.number_input("Age", min_value=20, max_value=60, value=30)
            monthlyincome = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
            yearsatcompany = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
            totalworkingyears = st.number_input("Total Working Years", min_value=0, max_value=50, value=10)

            # Build input DataFrame with raw values (no encoding needed)
            input_df = pd.DataFrame([{
                "Gender": gender,
                "Department": department,
                "JobRole": jobrole,
                "OverTime": overtime,
                "JobSatisfaction": jobsatisfaction,
                "Age": age,
                "MonthlyIncome": monthlyincome,
                "YearsAtCompany": yearsatcompany,
                "TotalWorkingYears": totalworkingyears
            }])

            if st.button("Run Prediction"):
                prediction = attrition_model.predict(input_df)[0]
                prob = attrition_model.predict_proba(input_df)[0][1]

                if prediction == "Yes" if prob > 0.5 else "No":
                    st.error(f"⚠️ Employee is likely to leave. (Prob: {prob:.2f})")
                else:
                    st.success(f"✅ Employee is likely to stay. (Prob: {prob:.2f})")

        # -------------------------
        # 2. Performance Prediction
        # -------------------------
        elif choice == "Performance Prediction":
            st.subheader("📈 Performance Prediction")

            jobrole = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician", "Manager",
                "Manufacturing Director", "Healthcare Representative", "Sales Representative",
                "Human Resources", "Other"
            ])
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            jobinvolvement = st.selectbox("Job Involvement", ["Low","Medium","High","Very High"])
            worklifebalance = st.selectbox("Work Life Balance", ["Bad","Good","Better","Best"])
            jobsatisfaction = st.selectbox("Job Satisfaction", ["Low","Medium","High","Very High"])
            environmentsatisfaction = st.selectbox("Environment Satisfaction", ["Low","Medium","High","Very High"])
            trainingtimes = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
            yearsatcompany = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
            yearsincurrentrole = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
            yearswithcurrmanager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=2)
            totalworkingyears = st.number_input("Total Working Years", min_value=0, max_value=50, value=10)

            input_df = pd.DataFrame([{
                "JobRole": jobrole,
                "OverTime": overtime,
                "JobInvolvement": jobinvolvement,
                "WorkLifeBalance": worklifebalance,
                "JobSatisfaction": jobsatisfaction,
                "EnvironmentSatisfaction": environmentsatisfaction,
                "TrainingTimesLastYear": trainingtimes,
                "YearsAtCompany": yearsatcompany,
                "YearsInCurrentRole": yearsincurrentrole,
                "YearsWithCurrManager": yearswithcurrmanager,
                "TotalWorkingYears": totalworkingyears
            }])

            if st.button("Run Prediction"):
                prediction = performance_model.predict(input_df)[0]
                prob = performance_model.predict_proba(input_df)[0][1]

                if prediction == "Outstanding" if prob > 0.5 else "Excellent":
                    st.success(f"☑️ Predicted Performance Rating: 4 (Outstanding). Probability: {prob:.2f}")
                else:
                    st.success(f"✅ Predicted Performance Rating: 3 (Excellent). Probability: {prob:.2f}")


        # -------------------------
        # 3. Promotion Likelihood
        # -------------------------
        elif choice == "Promotion Prediction":
            st.subheader("🚀 Promotion Prediction")

            joblevel = st.selectbox("Job Level", ["Entry_Level", "Junior_Level", "Mid_Level", "Senior_Level", "Executive_Level"])
            totalworkingyears = st.number_input("Total Working Years", min_value=0, max_value=50, value=10)
            yearsincurrentrole = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
            performancerating = st.selectbox("Performance Rating", ["Excellent","Outstanding"])
            education = st.selectbox("Education", ["High School", "Diploma", "Graduate", "Postgraduate", "Doctorate"])
            trainingtimes = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
            yearsatcompany = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
            yearswithcurrmanager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=2)

            input_df = pd.DataFrame([{
                "JobLevel": joblevel,
                "TotalWorkingYears": totalworkingyears,
                "YearsInCurrentRole": yearsincurrentrole,
                "PerformanceRating": performancerating,
                "Education": education,
                "TrainingTimesLastYear": trainingtimes,
                "YearsAtCompany": yearsatcompany,
                "YearsWithCurrManager": yearswithcurrmanager
            }])

            if st.button("Run Prediction"):
                prediction = promotion_model.predict(input_df)[0]
                prob = promotion_model.predict_proba(input_df).max()

                if prediction == "Soon":
                    st.success(f"✅ Likely to get promoted soon! (Confidence: {prob:.2f})")
                elif prediction == "Later":
                    st.warning(f"⏳ Promotion expected in the mid-term. (Confidence: {prob:.2f})")
                else:
                    st.error(f"❌ Promotion unlikely in near future. (Confidence: {prob:.2f})")





