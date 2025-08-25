# 👥 Employee Attrition Analysis and Prediction

## 📌 Project Overview
Employee turnover poses a major challenge for organizations, leading to higher costs, reduced productivity, and workplace disruptions.  
This project analyzes employee data to identify the **key drivers of attrition** and builds **predictive models** to support HR teams with data-driven retention strategies.

The project includes:

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Statistical Analysis  
- Machine Learning Model Development  
- Deployment via an **interactive Streamlit Dashboard**  

---

## 🎯 Objectives
- Identify factors influencing employee attrition.  
- Build machine learning models to predict:  
  - **Attrition Risk** (Who is likely to leave?)  
  - **Performance Rating** (Employee performance evaluation)  
  - **Promotion Likelihood** (Who is likely to get promoted soon?)  
- Provide actionable insights via interactive dashboards for HR decision-making.  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Imbalanced-learn  
- **Visualization & Dashboard:** Streamlit, Plotly, Seaborn  
- **Machine Learning Models:** Logistic Regression, Random Forest, Naive Bayes, Decision Trees  
- **Other Tools:** Jupyter Notebook, Git, Lottie animations  

---

## 📂 Repository Structure

-**1.Data_Cleaning.ipynb** # Data preprocessing & cleaning
- **2.Data_Visualization.ipynb** # Exploratory Data Analysis (EDA)
-**3.Stastical_Analysis.ipynb** # Hypothesis testing & statistical analysis
-**4.Data_Modeling.ipynb** # ML model training & evaluation
-**Employee_dashboard.py** # Streamlit dashboard app
-models/ # Saved trained models (Pickle/Joblib)
-data/ # Raw & processed datasets
-README.md # Project description (this file)

---

## 📊 Dataset
The dataset contains **35 features** related to employees’ demographics, job roles, salaries, performance, and work-life balance.  

Key features include:  
- **Age, Gender, Department, JobRole**  
- **MonthlyIncome, YearsAtCompany, JobSatisfaction**  
- **OverTime, PerformanceRating, WorkLifeBalance**  
- **Attrition (Target Variable: Yes/No)**  

---

## 🚀 Approach
1. **Data Collection & Cleaning**  
   - Handle missing values, outliers, categorical encoding, and normalization.  
   - Convert categorical features like JobRole, Education, etc. into numerical form.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize attrition patterns across departments, gender, age groups, and income levels.  
   - Identify correlations between features (e.g., OverTime, JobSatisfaction, MonthlyIncome).  

3. **Statistical Analysis**  
   - Conduct hypothesis testing (Chi-square, ANOVA, correlation tests).  
   - Identify significant predictors of attrition.  

4. **Model Building**  
   - Implement **Random Forest, Logistic Regression, Naive Bayes**.  
   - Handle class imbalance using **SMOTE**.  
   - Evaluate models with Accuracy, Precision, Recall, F1-Score, AUC-ROC.  

5. **Deployment**  
   - A **Streamlit app** was developed (`Employee_dashboard.py`) with:  
     - **Employee Records Viewer** (filter by department, gender, age, etc.)  
     - **Analytics Dashboard** (attrition trends, satisfaction, risk scoring)  
     - **Predictive Models** (Attrition, Performance, Promotion prediction)  

---

## 📈 Results
- Achieved **>85% accuracy** in attrition prediction.  
- Identified key drivers of attrition:  
  - Low Job Satisfaction  
  - High Overtime  
  - Longer Commute Distance  
  - Low Salary Growth & Career Progression  
- Dashboard enables HR to filter records, monitor KPIs, and predict employee outcomes.  

---

## 🖥️ Streamlit Dashboard
The interactive dashboard provides:  
✅ **Employee Records:** Filter by Department, Gender, Marital Status, Job Level, Age, Attrition.  
✅ **Analytics & Insights:** Attrition rate, Job Satisfaction, Performance, Work-Life Balance.  
✅ **Predictive Models:**  
   - ⚠️ Attrition Risk Prediction  
   - 📈 Performance Prediction  
   - 🚀 Promotion Likelihood  

## Run the Streamlit app

streamlit run Employee_dashboard.py

---

## 📊 Evaluation Metrics

**Accuracy**

**Precision & Recall**

**F1-Score**

**AUC-ROC Curve**

**Confusion Matrix**

---

## 💡 Business Impact

**Attrition Reduction**: Identify high-risk employees and apply retention strategies.

**Cost Optimization**: Reduce hiring and training costs.

**Workforce Planning**: Improve long-term employee engagement & satisfaction.

