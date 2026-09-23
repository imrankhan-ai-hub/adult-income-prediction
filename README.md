# Adult Income Prediction

##  Project Overview

This project predicts whether a person's annual income is **≤50K or >50K** using demographic, employment, education, and financial information from the **Adult Income Dataset**.

The main goal is to build a practical machine-learning classification system that handles real-world issues such as **missing values, categorical data, class imbalance, and model deployment**.

## Dataset

The dataset contains information such as:

* Age
* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Capital Gain/Loss
* Hours per Week
* Native Country

**Target:** `income`

* `0` → ≤50K
* `1` → >50K

## What I Did

1. Loaded and inspected the dataset.
2. Cleaned column names and categorical values.
3. Converted `?` values into missing values.
4. Handled missing numeric values using training-set medians.
5. Handled missing categorical values using training-set modes.
6. Removed duplicate records.
7. Converted the target variable into binary values.
8. Split the data using a **stratified train/test split**.
9. Converted categorical features using **One-Hot Encoding**.
10. Standardized numerical features.
11. Addressed class imbalance using **Balanced Random Forest (`class_weight="balanced"`)**.
12. Evaluated the model using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

##  Model

The final model is:

**Balanced Random Forest Classifier**

```text
n_estimators = 200
class_weight = "balanced"
random_state = 42
n_jobs = -1
```

### Why Balanced Random Forest?

The income classes are imbalanced, with the `≤50K` class considerably larger than the `>50K` class. Using balanced class weights gives greater importance to the minority class instead of allowing the model to focus mainly on the majority class.

Random Forest was also selected because it can handle nonlinear relationships and interactions between different demographic and employment features effectively.

## Saved Model Components

The trained model and preprocessing components were saved separately:

```text
adult_income_model/
├── random_forest_balanced.pkl
├── onehot_encoder.pkl
├── scaler.pkl
├── numeric_columns.pkl
├── categorical_columns.pkl
├── encoded_columns.pkl
├── numeric_medians.pkl
├── categorical_modes.pkl
└── training_feature_order.pkl
```

These files allow the same preprocessing and trained model to be reused for new predictions.

## 🖥️ Prediction

A command-line prediction program is included:

```text
predict.py
```

The user enters the original 14 input features. The program automatically:

**Input → Missing-value handling → Encoding → Scaling → Feature ordering → Balanced Random Forest → Prediction**

Example output:

```text
Predicted Income: >50K

Probability ≤50K: XX.XX%
Probability >50K: XX.XX%
```

##  Key Findings

* The dataset contains both numerical and categorical features.
* Several categorical columns contain missing values represented by `?`.
* The target variable is imbalanced.
* Proper preprocessing was required before model training.
* One-hot encoding converted categorical variables into numerical features.
* The final model uses **Balanced Random Forest** to account for class imbalance.
* The complete preprocessing and prediction process can be reused through the saved model components and CLI application.

```

## 🎯 Conclusion

This project demonstrates a complete machine-learning workflow from **data cleaning and exploratory analysis to preprocessing, imbalance handling, model training, evaluation, model saving, and CLI deployment**.

The final system can take raw user information and predict the expected income category using the trained **Balanced Random Forest** model.
