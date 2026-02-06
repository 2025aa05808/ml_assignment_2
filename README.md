# 📊 Machine Learning Assignment 2 – Classification Models & Deployment

## 1. Problem Statement
The objective of this assignment is to design, implement, and deploy multiple machine learning classification models on a real-world dataset. The project demonstrates the complete end-to-end machine learning workflow, including data preprocessing, model training, evaluation using standard classification metrics, and deployment through an interactive Streamlit web application.

---

## 2. Dataset Description  **[1 Mark]**
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository
- **Problem Type:** Binary Classification
- **Number of Instances:** 569
- **Number of Features:** 30 numerical features
- **Target Variable:**  
  - `diagnosis = 1` → Malignant  
  - `diagnosis = 0` → Benign
- **Description:**  
  The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses.

---

## 3. Models Used and Evaluation Metrics  **[6 Marks]**

### Implemented Models
1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Metrics
Accuracy, AUC, Precision, Recall, F1 Score, MCC

### Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|---------|-----|----------|--------|----|-----|
| Logistic Regression | 0.97 | 0.99 | 0.97 | 0.98 | 0.97 | 0.94 |
| Decision Tree | 0.93 | 0.94 | 0.92 | 0.94 | 0.93 | 0.86 |
| KNN | 0.96 | 0.98 | 0.96 | 0.97 | 0.96 | 0.92 |
| Naive Bayes | 0.94 | 0.97 | 0.94 | 0.95 | 0.94 | 0.88 |
| Random Forest | 0.98 | 0.99 | 0.98 | 0.99 | 0.98 | 0.96 |
| XGBoost | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.97 |

---

## 4. Model Observations  **[3 Marks]**

| Model | Observation |
|------|------------|
| Logistic Regression | Strong performance due to linear separability |
| Decision Tree | Slight overfitting |
| KNN | Sensitive to scaling |
| Naive Bayes | Assumption-based but stable |
| Random Forest | Excellent generalization |
| XGBoost | Best overall performance |

---

## 5. Streamlit Application
The Streamlit app allows:
- CSV test data upload  
- Model selection  
- Metric visualization  
- Confusion matrix & classification report  
- Downloadable sample test dataset  

---

## 6. Deployment
The application is deployed on Streamlit Community Cloud and accessible via a public link.

---

## 7. Compliance
All assignment requirements are satisfied, including dataset constraints, model implementation, metrics, and deployment.
