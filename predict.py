

import pandas as pd
import numpy as np
import joblib
import os




BASE_DIR = "adult_income_model"




model = joblib.load(
    os.path.join(
        BASE_DIR,
        "random_forest_balanced.pkl"
    )
)



encoder = joblib.load(
    os.path.join(
        BASE_DIR,
        "onehot_encoder.pkl"
    )
)


# =========================================================
# LOAD SCALER
# =========================================================

scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "scaler.pkl"
    )
)


# =========================================================
# LOAD MISSING VALUE INFORMATION
# =========================================================

numeric_medians = joblib.load(
    os.path.join(
        BASE_DIR,
        "numeric_medians.pkl"
    )
)

categorical_modes = joblib.load(
    os.path.join(
        BASE_DIR,
        "categorical_modes.pkl"
    )
)


# =========================================================
# LOAD FEATURE INFORMATION
# =========================================================

numeric_columns = joblib.load(
    os.path.join(
        BASE_DIR,
        "numeric_columns.pkl"
    )
)

categorical_columns = joblib.load(
    os.path.join(
        BASE_DIR,
        "categorical_columns.pkl"
    )
)

encoded_columns = joblib.load(
    os.path.join(
        BASE_DIR,
        "encoded_columns.pkl"
    )
)

training_feature_order = joblib.load(
    os.path.join(
        BASE_DIR,
        "training_feature_order.pkl"
    )
)


# =========================================================
# INPUT FUNCTIONS
# =========================================================

def get_numeric_input(prompt):

    while True:

        value = input(prompt).strip()

        if value == "":
            return np.nan

        try:
            return float(value)

        except ValueError:
            print("Please enter a valid number.")


def get_text_input(prompt):

    value = input(prompt).strip()

    if value == "":
        return np.nan

    return value


# =========================================================
# HEADER
# =========================================================

print()
print("=" * 60)
print("             ADULT INCOME PREDICTION")
print("=" * 60)
print()


# =========================================================
# GET USER INPUT
# =========================================================

data = {}


data["age"] = get_numeric_input(
    "Age: "
)

data["workclass"] = get_text_input(
    "Workclass: "
)

data["fnlwgt"] = get_numeric_input(
    "Final Weight (fnlwgt): "
)

data["education"] = get_text_input(
    "Education: "
)

data["education_num"] = get_numeric_input(
    "Education Number: "
)

data["marital_status"] = get_text_input(
    "Marital Status: "
)

data["occupation"] = get_text_input(
    "Occupation: "
)

data["relationship"] = get_text_input(
    "Relationship: "
)

data["race"] = get_text_input(
    "Race: "
)

data["sex"] = get_text_input(
    "Sex: "
)

data["capital_gain"] = get_numeric_input(
    "Capital Gain: "
)

data["capital_loss"] = get_numeric_input(
    "Capital Loss: "
)

data["hours_per_week"] = get_numeric_input(
    "Hours Per Week: "
)

data["native_country"] = get_text_input(
    "Native Country: "
)


# =========================================================
# CREATE DATAFRAME
# =========================================================

input_df = pd.DataFrame([data])


# =========================================================
# HANDLE MISSING NUMERICAL VALUES
# =========================================================

for column in numeric_columns:

    if pd.isna(input_df.loc[0, column]):

        input_df.loc[0, column] = numeric_medians[column]


# =========================================================
# HANDLE MISSING CATEGORICAL VALUES
# =========================================================

for column in categorical_columns:

    if pd.isna(input_df.loc[0, column]):

        input_df.loc[0, column] = categorical_modes[column]


# =========================================================
# ONE-HOT ENCODE CATEGORICAL FEATURES
# =========================================================

encoded_cat = encoder.transform(
    input_df[categorical_columns]
)

encoded_cat = pd.DataFrame(
    encoded_cat,
    columns=encoded_columns,
    index=input_df.index
)


# =========================================================
# NUMERICAL FEATURES
# =========================================================

numeric_data = input_df[
    numeric_columns
].copy()


# =========================================================
# SCALE NUMERICAL FEATURES
# =========================================================

numeric_data[numeric_columns] = scaler.transform(
    numeric_data[numeric_columns]
)


# =========================================================
# COMBINE NUMERICAL + CATEGORICAL
# =========================================================

final_input = pd.concat(
    [
        numeric_data,
        encoded_cat
    ],
    axis=1
)



final_input = final_input[
    training_feature_order
]


# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(
    final_input
)[0]


probability = model.predict_proba(
    final_input
)[0]


# =========================================================
# DISPLAY RESULT
# =========================================================

print()
print("=" * 60)
print("                 PREDICTION RESULT")
print("=" * 60)


if prediction == 1:

    print("Predicted Income : >50K")

else:

    print("Predicted Income : <=50K")


print(
    f"<=50K Probability : {probability[0] * 100:.2f}%"
)

print(
    f">50K Probability  : {probability[1] * 100:.2f}%"
)

print("=" * 60)