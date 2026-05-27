import numpy as np

from src.preprocess import (
    fetch_county_dataset,
    split_data,
    get_feature_types,
    build_preprocessor
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def train_elastic_net(X_train, y_train, X_test, y_test, preprocessor):

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
            alphas=np.logspace(-3, 1, 20),
            cv=5,
            random_state=42,
            max_iter=10000
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, y_pred, {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


# =========================
# TEST
# =========================

df = fetch_county_dataset(extended=False)

df["UninsuredRate"] = (
    df["Count_Household_NoHealthInsurance"] /
    df["Count_Person"]
)

df = df.dropna(subset=["UninsuredRate"])

if "place" in df.columns:
    df = df.drop(columns=["place"])

X_train, X_test, y_train, y_test = split_data(df, "UninsuredRate")

numeric_features, categorical_features = get_feature_types(X_train)

preprocessor = build_preprocessor(
    numeric_features,
    categorical_features
)

model, y_pred, metrics = train_elastic_net(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor
)

print(metrics)