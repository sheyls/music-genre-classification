from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import contextlib


class MLAlgorithms:
    def __init__(self, X, feature_names=None):
        self.X = X
        # Check if feature_names is not None and contains elements
        self.feature_names = list(feature_names) if feature_names is not None else [f"Feature {i}" for i in range(X.shape[1])]


class UnsupervisedAlgorithms(MLAlgorithms):
    def __init__(self, X, feature_names=None):
        super().__init__(X, feature_names)

    def evaluate_clustering(self, clusters, model_name):
        """Evaluates a clustering model with multiple metrics."""
        silhouette_avg = silhouette_score(self.X, clusters)
        db_score = davies_bouldin_score(self.X, clusters)
        ch_score = calinski_harabasz_score(self.X, clusters)
        print(f"--- {model_name} Clustering Evaluation ---")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Davies-Bouldin Index: {db_score:.3f}")
        print(f"Calinski-Harabasz Score: {ch_score:.3f}")
        return silhouette_avg, db_score, ch_score

    def visualize_clusters_2d(self, features_pca, clusters, title="Cluster Visualization"):
        """Visualizes clusters in 2D using PCA."""
        plt.figure(figsize=(10, 7))
        plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap="viridis", s=50, alpha=0.7)
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster")
        plt.grid(True)
        output_path = os.path.join("results/clustering/partitional", f"{title.replace(' ', '_')}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def cluster_size_distribution(self, clusters):
        """Visualizes the distribution of cluster sizes."""
        unique, counts = np.unique(clusters, return_counts=True)
        plt.figure(figsize=(8, 5))
        plt.bar(unique, counts, color='skyblue')
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Points")
        output_path = os.path.join("results/clustering/partitional", "Cluster_Size_Distribution.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def feature_importance_analysis(self, clusters):
        """Calculates the importance of features in forming clusters and saves the results."""
        cluster_means = pd.DataFrame(self.X, columns=self.feature_names).groupby(clusters).mean()
        global_mean = pd.DataFrame(self.X, columns=self.feature_names).mean(axis=0)
        importance = (cluster_means - global_mean).abs().mean().sort_values(ascending=False)
        print("--- Feature Importance ---")
        print(importance)

        plt.figure(figsize=(10, 6))
        importance.plot(kind='bar', color='skyblue')
        plt.title("Feature Importance in Clustering")
        plt.ylabel("Average Deviation from Global Mean")
        plt.xlabel("Feature")
        plt.grid(True)
        output_path = os.path.join("results/clustering/partitional", "Feature_Importance.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def cluster_characteristics(self, clusters, df):
        """Generates descriptive statistics and visualizes characteristics by cluster."""
        df['Cluster'] = clusters
        print("--- Cluster Statistics ---")
        print(df.groupby('Cluster').describe())

        for feature in self.feature_names:
            formatted_feature = feature.replace(" ", "_").replace("/", "_")
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='Cluster', y=feature, data=df)
            plt.title(f"{feature} by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel(feature)
            plt.grid(True)
            output_path = os.path.join(f"results/clustering/partitional/{formatted_feature}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")


class PartitionalClustering(UnsupervisedAlgorithms):
    def __init__(self, X, feature_names=None):
        super().__init__(X, feature_names)

    def determine_optimal_clusters_elbow(self, max_clusters=15):
        """Calculates WCSS to find the optimal number of clusters using the Elbow Method."""
        print("Calculating WCSS for Elbow Method...")
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(self.X)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")
        plt.grid(True)
        output_path = os.path.join("results/clustering/partitional", "Elbow_Method.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

        deltas = [wcss[i] - wcss[i + 1] for i in range(len(wcss) - 1)]
        second_derivative = [deltas[i] - deltas[i + 1] for i in range(len(deltas) - 1)]
        optimal_clusters = second_derivative.index(max(second_derivative)) + 2
        print(f"Optimal number of clusters determined: {optimal_clusters}")
        return optimal_clusters

    def kmeans_clustering(self, n_clusters=3):
        """Runs K-Means Clustering."""
        print("Running K-Means Clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.X)
        self.evaluate_clustering(clusters, "K-Means")
        return kmeans, clusters


if __name__ == '__main__':
    df = pd.read_csv('dataset/clusters_standard_data.csv')
    features = df.drop(columns=['Class', 'Artist Name', 'Track Name']).values
    feature_names = df.drop(columns=['Class', 'Artist Name', 'Track Name']).columns

    path = "results/clustering/partitional"
    os.makedirs(path, exist_ok=True)

    log_file = os.path.join(path, "output_log.txt")
    with open(log_file, "w") as log, contextlib.redirect_stdout(log):
        partitional_clustering = PartitionalClustering(features, feature_names)
        optimal_clusters = partitional_clustering.determine_optimal_clusters_elbow()
        kmeans, base_clusters = partitional_clustering.kmeans_clustering(n_clusters=optimal_clusters)

        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        partitional_clustering.visualize_clusters_2d(features_pca, base_clusters, title="Base Clusters")

        partitional_clustering.cluster_size_distribution(base_clusters)
        partitional_clustering.cluster_characteristics(base_clusters, df)
        partitional_clustering.feature_importance_analysis(base_clusters)
