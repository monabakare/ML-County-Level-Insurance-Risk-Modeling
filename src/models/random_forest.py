import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.preprocess import (
    fetch_county_dataset,
    split_data,
    build_preprocessor,
    get_feature_types,
)


def train_random_forest(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train a RandomForestRegressor using a preprocessing pipeline.
    Expects raw X_train/X_test data; preprocessing happens inside the pipeline.
    """

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rf),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    return model, y_pred, metrics


# ============================================================
# TESTING / DEBUGGING ONLY
# ------------------------------------------------------------

if __name__ == "__main__":
    df = fetch_county_dataset(extended=True)

    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] / df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"])
    df = df.drop(columns=["place"], errors="ignore")

    target_column = "UninsuredRate"

    X_train, X_test, y_train, y_test = split_data(df, target_column)

    numeric_features, categorical_features = get_feature_types(X_train)

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model, y_pred, metrics = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
    )

    print("\nRandom Forest Results")
    print("---------------------")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R^2:  {metrics['r2']:.4f}")