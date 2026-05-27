"""
Preprocessing pipeline for county-level insurance risk modeling

Supports:
- Base dataset (simple features)
- Extended dataset (feature engineering for stronger models)
"""

import os
import numpy as np
import pandas as pd
from datacommons_client.client import DataCommonsClient

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from dotenv import load_dotenv
import os

load_dotenv()

# =========================
# DATA FETCHING + FEATURE ENGINEERING
# =========================
def fetch_county_dataset(extended=False):
    """
    Fetch county-level dataset from Data Commons.

    Args:
        extended (bool): If True, includes additional features + feature engineering

    Returns:
        pd.DataFrame
    """
    client = DataCommonsClient(
        api_key=os.getenv("DATA_COMMONS_API_KEY")
    )

    base_variables = [
        "Count_Household_NoHealthInsurance",
        "Count_Person",
        "Median_Income_Household",
        "UnemploymentRate_Person",
    ]

    extra_variables = [
        "PovertyRate_Person",
        "Count_Person_BachelorDegreeOrHigher",
        "Median_Age_Person",
        "PopulationDensity"
    ]

    variables = base_variables + extra_variables if extended else base_variables

    response = client.observation.fetch(
        variable_dcids=variables,
        entity_expression="country/USA<-containedInPlace+{typeOf:County}",
        date="latest"
    )

    records = response.to_observation_records().model_dump()
    df = pd.DataFrame(records)

    # reshape
    df = df.groupby(["entity", "variable"])["value"].mean().reset_index()
    df = df.pivot(index="entity", columns="variable", values="value").reset_index()
    df.rename(columns={"entity": "place"}, inplace=True)

    # =========================
    # FEATURE ENGINEERING (EXTENDED)
    # =========================
    if extended:
        place_str = df["place"].astype(str)
        place_suffix = place_str.str.split("/").str[-1]
        df["state_fips"] = place_suffix.str[:2]

        if {"Count_Person_BachelorDegreeOrHigher", "Count_Person"}.issubset(df.columns):
            df["BachelorDegreeRate"] = (
                df["Count_Person_BachelorDegreeOrHigher"] /
                df["Count_Person"].replace(0, np.nan)
            )

        if "Median_Income_Household" in df.columns:
            df["Log_Median_Income_Household"] = np.log1p(
                df["Median_Income_Household"].clip(lower=0)
            )

        if "PopulationDensity" in df.columns:
            df["Log_PopulationDensity"] = np.log1p(
                df["PopulationDensity"].clip(lower=0)
            )

        if {"UnemploymentRate_Person", "PovertyRate_Person"}.issubset(df.columns):
            df["EconomicDistressIndex"] = (
                df["UnemploymentRate_Person"] +
                df["PovertyRate_Person"]
            ) / 2.0

    return df


# =========================
# SAVE / LOAD
# =========================
def save_data(df, filename="county_dataset.csv"):
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, "..", "data", filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

    return filepath


def load_data(filename):
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, "..", "data", filename)
    return pd.read_csv(filepath)


# =========================
# TRAIN / TEST SPLIT
# =========================
def split_data(df, target_column):
    """
    Split data into train/test with optional stratification for continuous targets
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_labels = None
    if y.nunique(dropna=True) >= 10:
        try:
            stratify_labels = pd.qcut(y, q=10, duplicates="drop")
        except Exception:
            stratify_labels = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels
    )

    return X_train, X_test, y_train, y_test


# =========================
# FEATURE TYPES
# =========================
def get_feature_types(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numeric_features, categorical_features


# =========================
# PREPROCESSOR
# =========================
def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


# =========================
# FULL PREPROCESS PIPELINE
# =========================
def preprocess_data(df, target_column):
    """
    Split data and apply preprocessing pipeline
    """
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    numeric_features, categorical_features = get_feature_types(X_train)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


# =========================
# TEST / DEBUG
# =========================
if __name__ == "__main__":
    df = fetch_county_dataset(extended=True)

    if "place" in df.columns:
        df = df.drop(columns=["place"])

    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] /
        df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"])

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df, "UninsuredRate"
    )

    print("Processed shape:", X_train.shape)