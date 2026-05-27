# centralized evaluation + visualization script
# runs regression, classification, clustering, cross-validation,
# and saves plots to docs/figures/

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

from sklearn.pipeline import Pipeline

from src.preprocess import (
    build_preprocessor,
    fetch_county_dataset,
    get_feature_types,
    split_data,
)

from src.models.elastic_net import train_elastic_net
from src.models.random_forest import train_random_forest


# ============================================================
# FIGURES DIRECTORY
# ============================================================

def _figures_dir():
    base_dir = Path(__file__).resolve().parent

    figures = (
        base_dir / ".." / "docs" / "figures"
    ).resolve()

    figures.mkdir(parents=True, exist_ok=True)

    return figures


# ============================================================
# DATA CLEANUP HELPERS
# ============================================================

def _regression_target_cleanup(df):

    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] /
        df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"])

    df = df.drop(
        columns=[
            "place",
            "Count_Household_NoHealthInsurance",
            "Count_Person",
        ],
        errors="ignore"
    )

    return df, "UninsuredRate"


def _classification_target_cleanup(df):

    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] /
        df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"])

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
        errors="ignore"
    )

    return df, "HighUninsuredRisk", threshold


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def _save_actual_vs_pred(y_true, y_pred, title, filename):

    path = _figures_dir() / filename

    plt.figure()

    plt.scatter(y_true, y_pred, alpha=0.7)

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--",
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[saved] {path}")


def _save_confusion_matrix(cm, title, filename):

    path = _figures_dir() / filename

    plt.figure()

    plt.imshow(cm, interpolation="nearest")

    plt.colorbar()

    plt.title(title)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0, 1], ["Low", "High"])
    plt.yticks([0, 1], ["Low", "High"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[saved] {path}")


# ============================================================
# METRIC HELPERS
# ============================================================

def _print_regression_metrics(name, y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Holdout Metrics")
    print("-" * 40)

    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R2:   {r2:.6f}")


def _run_regression_cv(name, estimator, X, y):

    scores = cross_validate(
        estimator,
        X,
        y,
        cv=5,
        scoring=(
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
            "r2",
        ),
        n_jobs=-1,
    )

    mae = -scores["test_neg_mean_absolute_error"]
    rmse = -scores["test_neg_root_mean_squared_error"]
    r2 = scores["test_r2"]

    print(f"\n{name} Cross-Validation (5-Fold)")
    print("-" * 40)

    print(f"MAE:  {mae.mean():.6f} +/- {mae.std():.6f}")
    print(f"RMSE: {rmse.mean():.6f} +/- {rmse.std():.6f}")
    print(f"R2:   {r2.mean():.6f} +/- {r2.std():.6f}")


# ============================================================
# CLASSIFICATION PIPELINES
# ============================================================

def _build_logistic_pipeline(X_train):

    numeric_features, categorical_features = (
        get_feature_types(X_train)
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])


def _build_rf_classifier_pipeline(X_train):

    numeric_features, categorical_features = (
        get_feature_types(X_train)
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


# ============================================================
# REGRESSION
# ============================================================

def run_regression_metrics():

    print("\n=== Regression Metrics + CV ===")

    # ElasticNet
    en_df, en_target = _regression_target_cleanup(
        fetch_county_dataset(extended=False)
    )

    en_X_train, en_X_test, en_y_train, en_y_test = split_data(
        en_df,
        en_target,
    )

    en_num, en_cat = get_feature_types(en_X_train)

    en_pre = build_preprocessor(
        en_num,
        en_cat,
    )

    en_model, en_pred, _ = train_elastic_net(
        en_X_train,
        en_y_train,
        en_X_test,
        en_y_test,
        en_pre,
    )

    _print_regression_metrics(
        "ElasticNet",
        en_y_test,
        en_pred,
    )

    _save_actual_vs_pred(
        en_y_test,
        en_pred,
        "ElasticNet: Actual vs Predicted",
        "elasticnet_cv_metrics.png",
    )

    _run_regression_cv(
        "ElasticNet",
        en_model,
        en_df.drop(columns=[en_target]),
        en_df[en_target],
    )

    # Random Forest
    rf_df, rf_target = _regression_target_cleanup(
        fetch_county_dataset(extended=True)
    )

    rf_X_train, rf_X_test, rf_y_train, rf_y_test = split_data(
        rf_df,
        rf_target,
    )

    rf_num, rf_cat = get_feature_types(rf_X_train)

    rf_pre = build_preprocessor(
        rf_num,
        rf_cat,
    )

    rf_model, rf_pred, _ = train_random_forest(
        rf_X_train,
        rf_y_train,
        rf_X_test,
        rf_y_test,
        rf_pre,
    )

    _print_regression_metrics(
        "RandomForestRegressor",
        rf_y_test,
        rf_pred,
    )

    _save_actual_vs_pred(
        rf_y_test,
        rf_pred,
        "Random Forest: Actual vs Predicted",
        "random_forest_cv_metrics.png",
    )

    _run_regression_cv(
        "RandomForestRegressor",
        rf_model,
        rf_df.drop(columns=[rf_target]),
        rf_df[rf_target],
    )


# ============================================================
# CLUSTERING
# ============================================================

def run_clustering_metrics():

    print("\n=== Clustering Metrics + Viz ===")

    df = fetch_county_dataset(extended=True)

    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] /
        df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"])

    df_cluster = df.drop(
        columns=[
            "UninsuredRate",
            "place",
            "state_fips",
        ],
        errors="ignore",
    )

    numeric_features, categorical_features = (
        get_feature_types(df_cluster)
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    X_processed = preprocessor.fit_transform(df_cluster)

    X_dense = (
        X_processed.toarray()
        if hasattr(X_processed, "toarray")
        else X_processed
    )

    # PCA before clustering
    # PCA before clustering
    n_components = min(10, X_dense.shape[1])

    pca = PCA(n_components=n_components)

    X_reduced = pca.fit_transform(X_dense)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(X_reduced)

    sil = silhouette_score(X_reduced, labels)
    cal = calinski_harabasz_score(X_reduced, labels)
    dav = davies_bouldin_score(X_reduced, labels)

    print(f"Silhouette Score:        {sil:.6f}")
    print(f"Calinski-Harabasz Score: {cal:.6f}")
    print(f"Davies-Bouldin Score:    {dav:.6f}")


# ============================================================
# CLASSIFICATION
# ============================================================

def run_classification_metrics():

    print("\n=== Classification Metrics + CV ===")

    raw_df = fetch_county_dataset(extended=True)

    cls_df, cls_target, threshold = (
        _classification_target_cleanup(raw_df)
    )

    print(
        f"Using HighUninsuredRisk threshold "
        f"at median UninsuredRate = {threshold:.6f}"
    )

    X_train, X_test, y_train, y_test = split_data(
        cls_df,
        cls_target,
    )

    logistic = _build_logistic_pipeline(X_train)

    rf_classifier = _build_rf_classifier_pipeline(X_train)

    for name, model in [
        ("LogisticRegression", logistic),
        ("RandomForestClassifier", rf_classifier),
    ]:

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        prob = model.predict_proba(X_test)[:, 1]

        print(f"\n{name} Holdout Metrics")
        print("-" * 40)

        print(f"Accuracy:  {accuracy_score(y_test, pred):.6f}")
        print(f"Precision: {precision_score(y_test, pred):.6f}")
        print(f"Recall:    {recall_score(y_test, pred):.6f}")
        print(f"F1:        {f1_score(y_test, pred):.6f}")
        print(f"ROC-AUC:   {roc_auc_score(y_test, prob):.6f}")

        cm = confusion_matrix(y_test, pred)

        suffix = (
            "logistic"
            if "Logistic" in name
            else "rf_classifier"
        )

        _save_confusion_matrix(
            cm,
            f"{name}: Confusion Matrix",
            f"{suffix}_confusion_matrix.png",
        )

        cv = cross_validate(
            model,
            cls_df.drop(columns=[cls_target]),
            cls_df[cls_target],
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            ),
            scoring=(
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ),
            n_jobs=-1,
        )

        print(f"\n{name} Cross-Validation (5-Fold)")
        print("-" * 40)

        for metric_name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ):

            arr = cv[f"test_{metric_name}"]

            print(
                f"{metric_name.upper():<10}: "
                f"{arr.mean():.6f} +/- {arr.std():.6f}"
            )


# ============================================================
# RUNNERS
# ============================================================

def run_all():

    print(
        "Starting full metrics + visualization + CV run...\n"
    )

    run_regression_metrics()
    run_clustering_metrics()
    run_classification_metrics()

    print("\nAll metrics finished.")


if __name__ == "__main__":

    # examples:
    # RUN_SECTION=regression python -m src.evaluate
    # RUN_SECTION=clustering python -m src.evaluate
    # RUN_SECTION=classification python -m src.evaluate

    section = os.getenv(
        "RUN_SECTION",
        "all"
    ).strip().lower()

    if section == "regression":
        run_regression_metrics()

    elif section == "clustering":
        run_clustering_metrics()

    elif section == "classification":
        run_classification_metrics()

    else:
        run_all()