# Heart Disease Prediction Using Deep Learning

This repository contains a Machine Learning and Deep Learning project for predicting heart disease using clinical patient data. The project compares multiple classifiers and applies advanced techniques such as SMOTE, data augmentation, ensemble learning, and regularization. A Gradio-based web interface is included for easy interaction with the trained models.

---

## Project Overview

Heart disease is one of the leading causes of death worldwide. Early and accurate diagnosis can significantly improve patient outcomes.  
This project aims to:
- Predict the presence of heart disease using machine learning
- Compare multiple models and evaluation metrics
- Improve performance using deep learning and ensemble methods
- Provide a user-friendly web interface for real-world usage

---

## Dataset

- **Dataset:** UCI Cleveland Heart Disease Dataset  
- **Samples:** 1025  
- **Features:** 13 clinical attributes  
- **Target:**  
  - `0` → No heart disease  
  - `1` → Presence of heart disease  

### Features include:
- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol  
- Fasting blood sugar  
- Maximum heart rate  
- Exercise-induced angina  
- ST depression  
- Thalassemia  
- Number of major vessels  

---

## Data Preprocessing

- No missing values detected
- Feature scaling using **StandardScaler**
- Train-test split (80% / 20%) with stratification
- **SMOTE** applied to handle class imbalance
- Tabular data augmentation using Gaussian noise

---

## Implemented Models

The following models were implemented and compared:

1. **Logistic Regression** (Baseline)
   - L2 Regularization
   - Max iterations: 1000

2. **Multi-Layer Perceptron (MLP)**
   - Hidden layers: 64, 32 neurons
   - ReLU activation
   - Adam optimizer

3. **Deep Neural Network (DNN)**
   - Layers: 64 → 32 → 16
   - Dropout (0.3)
   - L2 Regularization
   - Early Stopping
   - Adam optimizer

4. **Ensemble Model**
   - Voting classifier combining LR, MLP, and DNN

---

## Evaluation Metrics

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Results

| Model | Accuracy | Precision | Recall | F1-score | AUC |
|------|----------|-----------|--------|----------|-----|
| Logistic Regression | 82.44% | 78.05% | 91.43% | 84.21% | 82.21% |
| MLP | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Deep Neural Network | 98.54% | 100.00% | 97.14% | 98.55% | 98.57% |
| Ensemble Model | 98.54% | 100.00% | 97.14% | 98.55% | 98.57% |

The Deep Learning model achieved the best overall performance and outperformed existing literature on the same dataset.

---

## Web Interface (Gradio)

A **Gradio-based web application** was developed that allows users to:
- Select a trained model
- Input clinical data via a form
- Get real-time predictions with probability scores
- View model performance metrics

