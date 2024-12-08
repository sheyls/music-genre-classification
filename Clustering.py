from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import contextlib

from MLAlgorithms import MLAlgorithms

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


class UnsupervisedAlgorithms(MLAlgorithms):
    def __init__(self, X, y=None, X_validation=None, y_validation=None, feature_names=None):
        super().__init__(X, y, X_validation, y_validation, feature_names)

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
    def __init__(self, X, y=None, X_validation=None, y_validation=None, feature_names=None):
        super().__init__(X, y, X_validation, y_validation, feature_names)


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
   
    def get_closest_to_centroid(self, clusters, df, model):
        """
        Finds the song closest to the centroid of each cluster and returns its details.

        Args:
            clusters (array-like): Cluster labels for each instance.
            df (DataFrame): Original DataFrame, including Artist Name and Track Name.
            model: Clustering model (must have cluster_centers_ for K-Means).

        Returns:
            dict: Details of the song closest to the centroid of each cluster.
        """
        df['Cluster'] = clusters
        closest_points = {}

        if hasattr(model, "cluster_centers_"):  # Check if the model has explicit centroids (e.g., K-Means)
            for cluster_id, centroid in enumerate(model.cluster_centers_):
                cluster_data = df[df['Cluster'] == cluster_id]
                distances = np.linalg.norm(
                    cluster_data.drop(columns=['Cluster', 'Class', 'Artist Name', 'Track Name']).values - centroid, axis=1
                )
                closest_index = distances.argmin()
                closest_song = cluster_data.iloc[closest_index]
                closest_points[cluster_id] = {
                    'Artist Name': closest_song['Artist Name'],
                    'Track Name': closest_song['Track Name'],
                    'Distance to Centroid': distances[closest_index],
                    'Details': closest_song
                }
                print(f"Cluster {cluster_id}: Closest song to centroid:")
                print(f"  Artist: {closest_song['Artist Name']}")
                print(f"  Track: {closest_song['Track Name']}")
                print(f"  Distance to Centroid: {distances[closest_index]}")
        else:
            print("This model does not support explicit centroids. Use K-Means or similar.")

        return closest_points


    def get_instances_by_cluster(self, clusters, df, cluster_id, filter_conditions=None):
        """
        Filters instances of a specific cluster with options for additional conditions.
        Includes Artist Name and Track Name in the output.

        Args:
            clusters (array-like): Cluster labels for each instance.
            df (DataFrame): Original DataFrame containing the data.
            cluster_id (int): ID of the cluster to analyze.
            filter_conditions (dict, optional): Additional conditions in {column: value} format.

        Returns:
            DataFrame: Instances meeting the filter, including Artist Name and Track Name.
        """
        df['Cluster'] = clusters
        filtered_df = df[df['Cluster'] == cluster_id]

        # Apply additional filter conditions
        if filter_conditions:
            for column, value in filter_conditions.items():
                filtered_df = filtered_df[filtered_df[column] == value]

        # Select key columns for output
        output_columns = ['Artist Name', 'Track Name', 'Class', 'Cluster']
        filtered_output = filtered_df[output_columns]

        print(f"--- Instances in Cluster {cluster_id} ---")
        print(filtered_output)
        return filtered_output


    def calculate_class_percentage_per_cluster(self, clusters, df):
        """
        Calculates the percentage of each class's total instances that belong to each cluster.

        Args:
            clusters (array-like): Cluster labels for each instance.
            df (DataFrame): Original DataFrame, must include Class.

        Returns:
            dict: Dictionary with percentages by class and cluster.
        """
        df['Cluster'] = clusters
        total_class_counts = df['Class'].value_counts()  # Total instances per class in the entire dataset
        cluster_percentages = {}

        for cluster_id in np.unique(clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            class_counts = cluster_data['Class'].value_counts()  # Counts per class in the cluster
            cluster_percentages[cluster_id] = {}

            for class_label, count in class_counts.items():
                total_count_for_class = total_class_counts[class_label]
                percentage = (count / total_count_for_class) * 100
                cluster_percentages[cluster_id][class_label] = percentage

        # Display calculated percentages
        print("--- Class Percentages per Cluster ---")
        for cluster_id, percentages in cluster_percentages.items():
            print(f"Cluster {cluster_id}:")
            for class_label, percentage in percentages.items():
                print(f"  Class {class_label}: {percentage:.2f}% of total instances")

        return cluster_percentages


    def get_classes_and_counts_by_cluster(self, clusters, df):
        """
        Identifies classes (Class), counts the number of instances per class by cluster,
        and displays two representative songs (Track Name) and artists (Artist Name) per class.

        Args:
            clusters (array-like): Cluster labels for each instance.
            df (DataFrame): Original DataFrame, must include Class, Artist Name, and Track Name.

        Returns:
            dict: Unique classes and their counts per cluster, including two representative songs and artists.
        """
        df['Cluster'] = clusters
        cluster_class_counts = {}

        for cluster_id in np.unique(clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            class_counts = cluster_data['Class'].value_counts().to_dict()
            cluster_class_counts[cluster_id] = {'Class Counts': class_counts, 'Representative': {}}

            print(f"Cluster {cluster_id}:")
            for class_label, count in class_counts.items():
                # Select two representative songs and artists
                representative_songs = cluster_data[cluster_data['Class'] == class_label][['Artist Name', 'Track Name']].head(2)
                songs = representative_songs.values.tolist()
                cluster_class_counts[cluster_id]['Representative'][class_label] = [
                    {'Artist': artist, 'Track': track} for artist, track in songs
                ]

                print(f"  Class {class_label}: {count} instances")
                print(f"    Representative Songs:")
                for artist, track in songs:
                    print(f"      Artist: {artist}, Track: {track}")

        return cluster_class_counts

class HierarchicalClustering(UnsupervisedAlgorithms):

    def __init__(self, X, y=None, X_validation=None, y_validation=None, feature_names=None):
        super().__init__(X, y, X_validation, y_validation, feature_names)


    def hierarchical_clustering(self, n_clusters=3, linkage_method='ward'):
        """Performs Agglomerative Hierarchical Clustering."""
        print("Running Hierarchical Clustering...")
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        clusters = model.fit_predict(self.X)
        self.evaluate_clustering(clusters, "Hierarchical")
        return model, clusters

    def plot_dendrogram(self, method='ward', annotate=False, labels=None):
        """Plots and saves the dendrogram for hierarchical clustering."""
        from scipy.cluster.hierarchy import dendrogram, linkage
        print("Generating Dendrogram...")
        Z = linkage(self.X, method=method)
        plt.figure(figsize=(10, 7))
        dendrogram(
            Z,
            labels=labels if annotate else None,
            leaf_rotation=90,
            leaf_font_size=10,
        )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        output_path = os.path.join("results/clustering/hierarchical", "Dendrogram.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved dendrogram to: {output_path}")

    def determine_optimal_clusters_dendrogram_distances(self, max_clusters=15, method='ward'):
        """
        Visualizes the merge distances for hierarchical clustering within a limited number of clusters
        to help determine the optimal number of clusters.
        """

        print("Calculating distances for hierarchical clustering...")

        # Compute linkage matrix
        Z = linkage(self.X, method=method)

        # Extract merge distances
        merge_distances = Z[:, 2]  # Distances between clusters at each merge
        num_clusters = range(1, max_clusters + 1)  # Ascending order: 1, 2, ..., max_clusters

        # Limit the number of clusters for the plot
        if max_clusters > len(merge_distances):
            print(f"Warning: max_clusters ({max_clusters}) exceeds the number of merges.")
            max_clusters = len(merge_distances)

        limited_distances = merge_distances[-max_clusters:]  # Select the last 'max_clusters' distances

        # Plot the distances
        plt.figure(figsize=(10, 6))
        plt.plot(num_clusters, limited_distances[::-1], marker='o', linestyle='--', label='Merge Distances')
        plt.title("Cluster Merge Distances (Hierarchical Clustering)")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Merge Distance")
        plt.grid(True)

        # Save the plot
        output_path = os.path.join("results/clustering/hierarchical", "Dendrogram_Distances_Limited.png")
        plt.savefig(output_path)
        plt.show()
        print(f"Saved limited merge distance plot to: {output_path}")

        # Find the "elbow" point in the limited range
        # Identify where the largest drop in distances occurs
        deltas = np.diff(limited_distances[::-1])  # Differences between consecutive distances
        elbow_point = np.argmax(np.abs(deltas)) + 2  # +1 to account for cluster numbering
        print(f"Optimal number of clusters determined based on distances: {elbow_point}")
        return elbow_point


    def get_merge_details(self, method='ward'):
        """Extracts and prints merge distances and cluster sizes at each step."""
        from scipy.cluster.hierarchy import linkage
        print("\n--- Merge Details ---")
        Z = linkage(self.X, method=method)
        merge_distances = Z[:, 2]
        cluster_sizes = Z[:, 3]
        for i, (distance, size) in enumerate(zip(merge_distances, cluster_sizes)):
            print(f"Merge {i+1}: Distance = {distance:.3f}, Cluster Size = {int(size)}")

    def representative_instances(self, clusters, original_df):
        """Identifies representative instances closest to the cluster centroid."""
        representative_instances = {}
        for cluster_id in np.unique(clusters):
            cluster_data = self.X[clusters == cluster_id]
            centroid = cluster_data.mean(axis=0)
            distances = np.linalg.norm(cluster_data - centroid, axis=1)
            closest_index = np.argmin(distances)
            representative_row = original_df[clusters == cluster_id].iloc[closest_index]
            representative_instances[cluster_id] = {
                "Artist": representative_row['Artist Name'],
                "Track": representative_row['Track Name'],
                "Distance to Centroid": distances[closest_index],
            }
        print("\n--- Representative Instances ---")
        for cluster_id, info in representative_instances.items():
            print(f"Cluster {cluster_id}:")
            print(f"  Artist: {info['Artist']}, Track: {info['Track']}")
            print(f"  Distance to Centroid: {info['Distance to Centroid']:.4f}")
        return representative_instances

    def class_distribution_per_cluster(self, clusters, original_df):
        """
        Calculates the percentage distribution of each class across clusters.

        For each class, determines what percentage of its total instances belong to each cluster.
        """
        class_distributions = {}
        total_per_class = original_df['Class'].value_counts()  # Total instances per class

        for cls in total_per_class.index:
            cls_data = original_df[original_df['Class'] == cls]  # Filter data for the current class
            cluster_counts = cls_data['Cluster'].value_counts()  # Count instances of the class in each cluster
            percentages = (cluster_counts / total_per_class[cls]) * 100  # Calculate percentages
            class_distributions[cls] = percentages

        # Print results
        print("\n--- Class Distribution Across Clusters ---")
        for cls, percentages in class_distributions.items():
            print(f"Class {cls}:")
            for cluster_id, pct in percentages.items():
                print(f"  Cluster {cluster_id}: {pct:.2f}% of total instances in Class {cls}")

        return class_distributions


    def cluster_split_details(self, method='ward'):
        """Analyzes the split hierarchy to describe the evolution of clusters."""
        from scipy.cluster.hierarchy import linkage
        Z = linkage(self.X, method=method)
        print("\n--- Cluster Split Details ---")
        for i in range(len(Z)):
            cluster_1 = int(Z[i, 0])
            cluster_2 = int(Z[i, 1])
            size = int(Z[i, 3])
            distance = Z[i, 2]
            print(f"Merge {i+1}: Cluster {cluster_1} + Cluster {cluster_2} -> New Size = {size}, Distance = {distance:.3f}")

class ProbabilisticClustering(UnsupervisedAlgorithms):

    def __init__(self, X, y=None, X_validation=None, y_validation=None, feature_names=None):
        super().__init__(X, y, X_validation, y_validation, feature_names)


    def gmm_clustering(self, n_components=3):
        """Performs clustering using Gaussian Mixture Models."""
        print("Running Probabilistic Clustering (GMM)...")
        model = GaussianMixture(n_components=n_components, random_state=42)
        clusters = model.fit_predict(self.X)
        self.evaluate_clustering(clusters, "Probabilistic")
        return model, clusters

    def cluster_probabilities(self, gmm_model):
        """Calculates and saves cluster probabilities for each data point."""
        probabilities = gmm_model.predict_proba(self.X)
        df_probabilities = pd.DataFrame(probabilities, columns=[f"Cluster {i}" for i in range(probabilities.shape[1])])
        output_path = os.path.join("results/clustering/probabilistic", "Cluster_Probabilities.csv")
        df_probabilities.to_csv(output_path, index=False)
        print(f"Saved cluster probabilities to: {output_path}")
        return df_probabilities

    def cluster_characteristics(self, clusters, df):
        """Generates descriptive statistics for each cluster."""
        df['Cluster'] = clusters
        print("--- Cluster Characteristics ---")
        cluster_stats = df.groupby('Cluster').describe()
        print(cluster_stats)
        output_path = os.path.join("results/clustering/probabilistic", "Cluster_Characteristics.csv")
        cluster_stats.to_csv(output_path)
        print(f"Saved cluster characteristics to: {output_path}")
        return cluster_stats

    def feature_importance_analysis(self, gmm_model):
        """Analyzes feature importance based on the means of GMM components."""
        print("\n--- Feature Importance ---")
        cluster_means = pd.DataFrame(gmm_model.means_, columns=self.feature_names)
        global_mean = np.mean(self.X, axis=0)
        importance = (cluster_means - global_mean).abs().mean(axis=0).sort_values(ascending=False)
        print(importance)

        plt.figure(figsize=(10, 6))
        importance.plot(kind='bar', color='skyblue')
        plt.title("Feature Importance in Probabilistic Clustering")
        plt.ylabel("Average Deviation from Global Mean")
        plt.xlabel("Feature")
        plt.grid(True)
        output_path = os.path.join("results/clustering/probabilistic", "Feature_Importance.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved feature importance plot to: {output_path}")

    def cluster_class_distribution(self, clusters, original_df):
        """Calculates class distribution percentages for each cluster."""
        class_percentages = {}
        for cluster_id in np.unique(clusters):
            cluster_data = original_df[original_df['Cluster'] == cluster_id]
            class_counts = cluster_data['Class'].value_counts()
            percentages = (class_counts / class_counts.sum()) * 100
            class_percentages[cluster_id] = percentages
        print("\n--- Class Percentages per Cluster ---")
        for cluster_id, percentages in class_percentages.items():
            print(f"Cluster {cluster_id}:")
            for cls, pct in percentages.items():
                print(f"  Class {cls}: {pct:.2f}% of instances in this cluster")
        return class_percentages

    def cluster_probabilities_visualization(self, df_probabilities):
        """Visualizes cluster probabilities for interpretability."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_probabilities)
        plt.title("Distribution of Cluster Probabilities")
        plt.xlabel("Cluster")
        plt.ylabel("Probability")
        output_path = os.path.join("results/clustering/probabilistic", "Cluster_Probabilities_Boxplot.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved cluster probabilities visualization to: {output_path}")


if __name__ == '__main__':
    # Cargar datos
    df = pd.read_csv('dataset/clusters_standard_data.csv')
    features = df.drop(columns=['Class', 'Artist Name', 'Track Name']).values
    feature_names = df.drop(columns=['Class', 'Artist Name', 'Track Name']).columns

    # Crear carpetas para resultados
    path1 = "results/clustering/partitional"
    path2 = "results/clustering/hierarchical"
    path3 = "results/clustering/probabilistic"
    os.makedirs(path1, exist_ok=True)
    os.makedirs(path2, exist_ok=True)
    os.makedirs(path3, exist_ok=True)

    # Configuración de PCA para todas las visualizaciones
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # Partitional Clustering
    log_file = os.path.join(path1, "output_log.txt")
    with open(log_file, "w") as log, contextlib.redirect_stdout(log):
        partitional_clustering = PartitionalClustering(features, feature_names=feature_names)
        optimal_clusters = partitional_clustering.determine_optimal_clusters_elbow()
        kmeans_model, base_clusters = partitional_clustering.kmeans_clustering(n_clusters=optimal_clusters)

        # Visualización y análisis
        partitional_clustering.visualize_clusters_2d(features_pca, base_clusters, title="Base Clusters")
        partitional_clustering.cluster_size_distribution(base_clusters)
        partitional_clustering.cluster_characteristics(base_clusters, df)
        partitional_clustering.feature_importance_analysis(base_clusters)

        # Explicability
        cluster_class_counts = partitional_clustering.get_classes_and_counts_by_cluster(base_clusters, df)
        closest_songs = partitional_clustering.get_closest_to_centroid(base_clusters, df, kmeans_model)
        class_percentages = partitional_clustering.calculate_class_percentage_per_cluster(base_clusters, df)

    # Hierarchical Clustering
    log_file = os.path.join(path2, "output_log.txt")
    with open(log_file, "w") as log, contextlib.redirect_stdout(log):
        hierarchical_clustering = HierarchicalClustering(features, feature_names=feature_names)
        hierarchical_clustering.plot_dendrogram(method='ward')

        optimal_clusters = hierarchical_clustering.determine_optimal_clusters_dendrogram_distances()

        hierarchical_model, hierarchical_clusters = hierarchical_clustering.hierarchical_clustering(
            n_clusters=optimal_clusters, linkage_method='ward'
        )

        # Visualización y análisis
        hierarchical_clustering.visualize_clusters_2d(features_pca, hierarchical_clusters, title="Hierarchical Clusters")
        hierarchical_clustering.cluster_size_distribution(hierarchical_clusters)
        hierarchical_clustering.cluster_characteristics(hierarchical_clusters, df)

        # Explicability
        hierarchical_clustering.representative_instances(hierarchical_clusters, df)
        hierarchical_clustering.class_distribution_per_cluster(hierarchical_clusters, df)
        hierarchical_clustering.cluster_split_details(method='ward')
        
    # Probabilistic Clustering
    log_file = os.path.join(path3, "output_log.txt")
    with open(log_file, "w") as log, contextlib.redirect_stdout(log):

        # Cluster probabilities

        probabilistic_clustering = ProbabilisticClustering(features, feature_names=feature_names)
        gmm_model, probabilistic_clusters = probabilistic_clustering.gmm_clustering(n_components=3)

        # Visualización y análisis
        probabilistic_clustering.visualize_clusters_2d(features_pca, probabilistic_clusters, title="Probabilistic Clusters")
        probabilistic_clustering.cluster_size_distribution(probabilistic_clusters)
        probabilistic_clustering.cluster_characteristics(probabilistic_clusters, df)
        probabilistic_clustering.cluster_probabilities(gmm_model)

        df_probabilities = probabilistic_clustering.cluster_probabilities(gmm_model)

        # Explicability
        probabilistic_clustering.feature_importance_analysis(gmm_model)
        probabilistic_clustering.cluster_probabilities_visualization(df_probabilities)
        probabilistic_clustering.cluster_characteristics(probabilistic_clusters, df)
        probabilistic_clustering.cluster_class_distribution(probabilistic_clusters, df)
