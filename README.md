# Heart Disease Prediction

A machine learning project to predict the presence of heart disease using clinical data.  
This project uses the UCI Heart Disease dataset and applies classification algorithms to build a predictive model.

---

## 📌 Problem  
The goal is to build an ML model that can predict whether a person has heart disease based on health parameters such as age, blood pressure, cholesterol, ECG results, etc.

---

## 🧠 What’s Inside  
This project includes:
- Data exploration (EDA)
- Train/test split
- Model training and evaluation
- Comparison of models (e.g., Logistic Regression, Random Forest)
- Performance metrics (accuracy, recall, F1, ROC AUC)
- Confusion matrix and ROC curve visualization

---

## 📊 Results Summary

**Model used:** Logistic Regression  
**Cross-validated metrics:**
- Accuracy: 84.69%
- Precision: 82.07%
- Recall: 92.21%
- F1-Score: 86.73%

**Final model chosen:** Logistic Regression  
Because it gave the best balance of performance metrics for this dataset.

---

## 📋 How to Run This Project

1.
import pickle
import numpy as np

 Load the trained model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

Example input (single patient data)
sample = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

Prediction
prediction = model.predict(sample)
print("Prediction:", prediction)

2. Clone the repository  
```bash
git clone https://github.com/sakshamkumarsingh11/Heart-disease-prediction.git



