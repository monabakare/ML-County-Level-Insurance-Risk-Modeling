
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


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

