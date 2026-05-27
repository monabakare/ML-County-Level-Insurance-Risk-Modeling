

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def train_random_forest_classifier(X_train, y_train, X_test, y_test, preprocessor):
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf_classifier)
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return model, y_pred, y_prob, metrics

