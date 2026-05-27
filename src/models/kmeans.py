import sys
import os


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.preprocess import (
    fetch_county_dataset,
    build_preprocessor,
    get_feature_types,
)

def run_kmeans_clustering(X_processed, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_processed)

    print("\nKMeans Clustering")
    print("------------------")
    print("Inertia:", kmeans.inertia_)

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    plt.title(f"KMeans Clusters (k={n_clusters})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    return clusters, kmeans


def plot_elbow_method(X_processed):
    inertias = []
    k_values = range(2, 10)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_processed)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.show()

#test
if __name__ == "__main__":
    df = fetch_county_dataset()

    df["UninsuredRate"] = df["Count_Household_NoHealthInsurance"] / df["Count_Person"]

    df = df.dropna(subset=["UninsuredRate"])

    df_cluster = df.drop(columns=["UninsuredRate"], errors="ignore")

    numeric_features, categorical_features = get_feature_types(df_cluster)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_processed = preprocessor.fit_transform(df_cluster)

    plot_elbow_method(X_processed)

    clusters, kmeans = run_kmeans_clustering(X_processed, n_clusters=4)

    df_clustered = df.copy()
    df_clustered["Cluster"] = clusters
    
    print("\nCluster Summary:")
    print(df_clustered.groupby("Cluster")["UninsuredRate"].mean())
    score = silhouette_score(X_processed, clusters)
    print("Silhouette Score: {:.3f}".format(score))
    if score > 0.5:
        print("strong clustering")
    elif score > 0.25:
        print("moderate")
    else:
        print("weak clustering")