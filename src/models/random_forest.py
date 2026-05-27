import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



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


