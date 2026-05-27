import numpy as np

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
