# 🔍 Credit Card Fraud Detection

> Detecting fraudulent transactions in a dataset of 1.3M records using XGBoost and advanced feature engineering.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)](https://xgboost.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()

---

## 📋 Project Overview

Credit card fraud is a critical problem for financial institutions. This project builds a binary classification model to detect fraudulent transactions from a highly imbalanced dataset where only **0.58%** of transactions are fraudulent.

The main challenge is handling the **severe class imbalance** while maximising recall on the minority class (fraud), as missing a fraudulent transaction is far more costly than a false alarm.

---

## 📊 Results

| Metric   | Score    |
|----------|----------|
| AUC-ROC  | **0.97** |

---

## 🗂️ Dataset

| Set   | Rows      | Fraud rate |
|-------|-----------|------------|
| Train | 1,296,675 | 0.58%      |
| Test  | 555,719   | —          |

**Features include:** transaction amount, merchant category, customer location, merchant location, date/time, customer demographics.

> ⚠️ The dataset is not included in this repository due to its size.  
> Download it from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/).

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Fraud rate analysis by hour, day of week and month
- Geographic distribution of fraudulent transactions
- Amount distribution: fraud vs legitimate
- Demographic analysis (age, gender, city population)

### 2. Feature Engineering
| Feature | Description |
|---------|-------------|
| `hour_sin` / `hour_cos` | Cyclic encoding of transaction hour |
| `dow_sin` / `dow_cos` | Cyclic encoding of day of week |
| `geo_dist` | Haversine distance between customer and merchant |
| `lat_diff` / `lon_diff` | Absolute coordinate differences |
| `age` | Customer age at time of transaction |
| `gr_age` | Age group (binned) |
| `is_night` | 1 if transaction between 22h–5h |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_high_risk_cat` | 1 if merchant category is high-risk |

### 3. Preprocessing
- **Label Encoding** with custom unknown category handling for unseen test values
- **SMOTE** oversampling to address class imbalance

### 4. Modeling
- **XGBoost Classifier** with `scale_pos_weight` tuned to the class ratio (~171)
- Evaluation metric: AUC-ROC and AUC-PR

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Modeling:** XGBoost, Scikit-learn
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Imbalance:** imbalanced-learn (SMOTE)

---

## 📁 Repository Structure

```
fraud-detection/
├── Fraud_detection.ipynb   ← main notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Brahim-Bai/fraud-detection.git
cd fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook Fraud_detection.ipynb
```

---

## 👤 Author

**Brahim Baizam Ezzaki**  
Junior Data Scientist · Barcelona  
[LinkedIn](https://www.linkedin.com/in/brahim-baizam-ezzaki-486509223) · [GitHub](https://github.com/Brahim-Bai)
