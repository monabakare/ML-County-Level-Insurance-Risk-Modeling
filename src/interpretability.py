
import os
import pandas as pd

from src.preprocess import (
    fetch_county_dataset,
    split_data,
    build_preprocessor,
    get_feature_types,
)

from src.models.elastic_net import train_elastic_net
from src.models.random_forest import train_random_forest
from src.models.logistic_regression import train_logistic_regression
from src.models.rf_classifier import train_random_forest_classifier


def _regression_target_cleanup(df):
    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] / df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"]).copy()

    df = df.drop(
        columns=[
            "place",
            "Count_Household_NoHealthInsurance",
            "Count_Person",
        ],
        errors="ignore",
    )
    df = df.drop(columns=["state_fips"], errors="ignore")

    return df, "UninsuredRate"


def _classification_target_cleanup(df):
    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] / df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"]).copy()

    threshold = float(df["UninsuredRate"].median())

    df["HighUninsuredRisk"] = (
        df["UninsuredRate"] >= threshold
    ).astype(int)

    df = df.drop(
        columns=[
            "place",
            "Count_Household_NoHealthInsurance",
            "Count_Person",
            "UninsuredRate",
        ],
        errors="ignore",
    )
    df = df.drop(columns=["state_fips"], errors="ignore")
    return df, "HighUninsuredRisk", threshold


def _get_feature_names(preprocessor):
    """
    Extract feature names after preprocessing.
    Works with numeric + one-hot encoded categorical features.
    """

    numeric_features = list(preprocessor.transformers_[0][2])

    try:
        categorical_pipeline = preprocessor.transformers_[1][1]
        categorical_features = preprocessor.transformers_[1][2]

        encoder = categorical_pipeline.named_steps["encoder"]

        encoded_categorical_features = encoder.get_feature_names_out(
            categorical_features
        ).tolist()

    except Exception:
        encoded_categorical_features = []

    return numeric_features + encoded_categorical_features


def _print_top_coefficients(name, coef_df, top_n=10):
    print("\n" + "=" * 60)
    print(f"{name} — Top Coefficients")
    print("=" * 60)

    if coef_df.empty:
        print("No coefficients found.")
        return

    coef_df = coef_df.copy()
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

    print(
        coef_df.sort_values("abs_coefficient", ascending=False)
        .drop(columns=["abs_coefficient"])
        .head(top_n)
        .to_string(index=False)
    )


def _print_top_importances(name, importance_df, top_n=10):
    print("\n" + "=" * 60)
    print(f"{name} — Top Feature Importances")
    print("=" * 60)

    if importance_df.empty:
        print("No feature importances found.")
        return

    print(
        importance_df.sort_values("importance", ascending=False)
        .head(top_n)
        .to_string(index=False)
    )


def run_elasticnet_coefficients(top_n=10):
    df, target_column = _regression_target_cleanup(
        fetch_county_dataset(extended=False)
    )

    X_train, X_test, y_train, y_test = split_data(df, target_column)

    numeric_features, categorical_features = get_feature_types(X_train)

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model, _, _ = train_elastic_net(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
    )

    fitted_preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = _get_feature_names(fitted_preprocessor)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": regressor.coef_,
        }
    )

    coef_df = coef_df[coef_df["coefficient"] != 0]

    _print_top_coefficients(
        "ElasticNet",
        coef_df,
        top_n=top_n,
    )


def run_random_forest_importances(top_n=10):
    df, target_column = _regression_target_cleanup(
        fetch_county_dataset(extended=True)
    )

    X_train, X_test, y_train, y_test = split_data(df, target_column)

    numeric_features, categorical_features = get_feature_types(X_train)

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model, _, _ = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
    )

    fitted_preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = _get_feature_names(fitted_preprocessor)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": regressor.feature_importances_,
        }
    )

    _print_top_importances(
        "Random Forest Regressor",
        importance_df,
        top_n=top_n,
    )


def run_logistic_coefficients(top_n=10):
    df, target_column, threshold = _classification_target_cleanup(
        fetch_county_dataset(extended=True)
    )

    print(
        f"\nUsing HighUninsuredRisk threshold "
        f"at median UninsuredRate = {threshold:.6f}"
    )

    X_train, X_test, y_train, y_test = split_data(df, target_column)

    numeric_features, categorical_features = get_feature_types(X_train)

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model, _, _, _ = train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
    )

    fitted_preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = _get_feature_names(fitted_preprocessor)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": classifier.coef_[0],
        }
    )

    _print_top_coefficients(
        "Logistic Regression",
        coef_df,
        top_n=top_n,
    )


def run_random_forest_classifier_importances(top_n=10):
    df, target_column, threshold = _classification_target_cleanup(
        fetch_county_dataset(extended=True)
    )

    print(
        f"\nUsing HighUninsuredRisk threshold "
        f"at median UninsuredRate = {threshold:.6f}"
    )

    X_train, X_test, y_train, y_test = split_data(df, target_column)

    numeric_features, categorical_features = get_feature_types(X_train)

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model, _, _, _ = train_random_forest_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
    )

    fitted_preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = _get_feature_names(fitted_preprocessor)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    )

    _print_top_importances(
        "Random Forest Classifier",
        importance_df,
        top_n=top_n,
    )


def run_all():
    print("\nRunning model interpretability analysis...")

    run_elasticnet_coefficients()
    run_random_forest_importances()
    run_logistic_coefficients()
    run_random_forest_classifier_importances()

    print("\nInterpretability analysis finished.")


if __name__ == "__main__":
    section = os.getenv("SECTION", "all").strip().lower()

    if section == "elasticnet":
        run_elasticnet_coefficients()

    elif section == "rf_regressor":
        run_random_forest_importances()

    elif section == "logistic":
        run_logistic_coefficients()

    elif section == "rf_classifier":
        run_random_forest_classifier_importances()

    else:
        run_all()