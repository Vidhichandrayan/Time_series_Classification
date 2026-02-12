# Time Series Rare Event Classification

## 📌 Project Overview
This project focuses on building a robust machine learning pipeline for **tabular time-series classification** with severe class imbalance (~0.8% positive class).

The objective was to predict rare events using historical tabular data while ensuring:
- No data leakage
- Proper chronological validation
- Balanced model evaluation

---

## ⚠️ Key Challenges
- Highly imbalanced dataset (≈ 99% class 0, 1% class 1)
- Risk of time-series data leakage
- Need for proper threshold tuning
- Avoiding misleading accuracy metrics

---

## 🧠 Approach

### 1️⃣ Data Preparation
- Loaded parquet dataset
- Converted `Date` to datetime
- Sorted chronologically
- Verified missing values
- Converted target to integer

### 2️⃣ Feature Engineering
To capture temporal dependencies:

- Lag Features (t-1, t-2)
- Rolling Mean (3 & 5 window)
- Difference Features (momentum)
- Time-based features (month, weekday)

All features were created **before time split** to prevent leakage.

---

### 3️⃣ Validation Strategy
Used chronological split:

- First 80% → Training
- Last 20% → Validation

No random shuffle used.

---

### 4️⃣ Models Tried

- Random Forest (baseline)
- LightGBM
- Threshold tuning

---

## 📊 Model Performance

### 🔹 Random Forest (Baseline)
- Default F1: ~0.54
- After Threshold Tuning: **~0.60**

### 🔹 LightGBM
- High recall but low precision
- Lower overall F1 compared to RandomForest

Final selected model: **RandomForest with optimized threshold**

---

## 📈 Key Insights

- Accuracy is misleading for imbalanced datasets
- Threshold tuning significantly improves F1
- Feature engineering impacts performance more than model swapping
- Proper time-based validation is critical

---

## 🛠 Tech Stack
- Python
- Pandas
- Scikit-Learn
- LightGBM
- NumPy

---

## 🚀 Future Improvements
- More advanced lag structures
- Rolling standard deviation features
- TimeSeriesSplit cross-validation
- XGBoost experimentation
- Hyperparameter tuning

---

This project demonstrates a complete ML pipeline for time-series classification including:

✔ Proper validation strategy  
✔ Handling severe class imbalance  
✔ Threshold optimization  
✔ Feature engineering  
✔ Model comparison  

The focus was on building a **robust and realistic predictive system**

