from src.preprocess import (
    fetch_county_dataset,
    split_data,
    build_preprocessor,
    get_feature_types,
)

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

## just for testing; can be removed once Metrics implemented 
if __name__ == "__main__":
    df = fetch_county_dataset(extended=True)

    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] /
        df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"])

    threshold = df["UninsuredRate"].median()
    df["HighRisk"] = (df["UninsuredRate"] > threshold).astype(int)

    df = df.drop(columns=[
        "UninsuredRate",
        "Count_Household_NoHealthInsurance",
        "Count_Person"
    ])

    target_column = "HighRisk"

    X_train, X_test, y_train, y_test = split_data(df, target_column)

    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model, y_pred, y_prob, metrics = train_random_forest_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor
    )

    print("\nRandom Forest Classifier Results")
    print("--------------------------------")
    for key, value in metrics.items():
        print(f"{key}: {value}")