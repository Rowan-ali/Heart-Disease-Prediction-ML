# ❤️ Heart Disease Prediction using Machine Learning

An end-to-end Machine Learning project for predicting heart disease using tree-based ensemble models, Explainable AI (SHAP), hyperparameter optimization, cost-sensitive learning, and an interactive deployment built with Streamlit and FastAPI.

---

## 📌 Project Highlights

- End-to-end machine learning workflow
- Comprehensive exploratory data analysis (EDA)
- Advanced feature engineering (8 engineered clinical features)
- Outlier detection using the IQR method
- Baseline model comparison
- Hyperparameter optimization using RandomizedSearchCV
- SHAP Explainable AI (global & local explanations)
- Cost-sensitive learning for healthcare applications
- Stacking ensemble
- Interactive deployment using Streamlit & FastAPI

---

## 🩺 Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early prediction can support clinical decision-making and improve patient outcomes.

This project develops a complete machine learning pipeline to predict whether a patient has heart disease based on demographic information, clinical measurements, and engineered features.

---

## 📊 Dataset

This project uses the publicly available **Heart Disease Dataset**.

---

## ⚙️ Project Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Outlier Detection & Handling
5. Baseline Model Development
6. Hyperparameter Tuning
7. Model Comparison
8. Feature Importance Analysis
9. Explainable AI (SHAP)
10. Cost-Sensitive Learning
11. Stacking Ensemble
12. Interactive Deployment

---

## 🤖 Models Implemented

- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Stratified 5-Fold Cross Validation

---

## 🧠 Explainability

The project incorporates SHAP (SHapley Additive exPlanations) to improve model transparency through:

- SHAP Summary Plot
- SHAP Force Plot
- Individual Prediction Explanations

---

## 🚀 Deployment

The best-performing model was deployed using:

- Streamlit
- FastAPI

The deployed application includes:

- ❤️ Patient Risk Prediction
- 📊 Interactive Analytics Dashboard
- 📈 Monitoring Dashboard
- 📉 Probability Estimation
- 📋 Model Performance Metrics

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Matplotlib
- Streamlit
- FastAPI
- Joblib

---

## 📂 Repository Structure

```text
Heart-Disease-Prediction-ML/
│
├── data/
├── deployment/
├── notebook/
├── screenshots/
├── README.md
└── requirements.txt
```

---

## 🎥 Demo Video

A complete demonstration of the deployed application is available here:

**🔗 Demo Video:** 

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction-ML.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run deployment/app.py
```

Run the FastAPI server:

```bash
uvicorn deployment.api:app --reload
```

---

## 📚 Key Machine Learning Concepts

- Feature Engineering
- Ensemble Learning
- Hyperparameter Optimization
- Explainable AI (SHAP)
- Cost-Sensitive Learning
- Model Deployment
- REST API Development

---

## 👩‍💻 Author

**Your Name**

- LinkedIn: https://www.linkedin.com/in/rowan-ali-ibrahim-ali/
- GitHub: https://github.com/Rowan-ali

---

⭐ If you found this project interesting, consider giving the repository a star!
