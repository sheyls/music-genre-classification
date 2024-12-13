from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances
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
        """
        Evaluates a clustering model with multiple metrics.
        
        Parameters:
            clusters (array-like): Cluster labels for each data point.
            model_name (str): Name of the clustering model being evaluated.
        
        Returns:
            dict: A dictionary containing all evaluation metrics.
        """
        if len(np.unique(clusters)) <= 1:
            raise ValueError("Clustering must have at least two distinct clusters.")
        
        metrics = {}
        
        # Silhouette Score
        metrics["Silhouette Score"] = silhouette_score(self.X, clusters)
        
        # Davies-Bouldin Index
        metrics["Davies-Bouldin Index"] = davies_bouldin_score(self.X, clusters)
        
        # Calinski-Harabasz Score
        metrics["Calinski-Harabasz Score"] = calinski_harabasz_score(self.X, clusters)
        
        # Cohesion and Separation
        def compute_cohesion_separation(X, labels):
            clusters = np.unique(labels)
            cohesion = []
            separation = []
            
            # Calculate cohesion (intra-cluster distances)
            for cluster in clusters:
                points_in_cluster = X[labels == cluster]
                distances = pairwise_distances(points_in_cluster)
                cohesion.append(np.mean(distances))
            
            # Calculate separation (distances between centroids)
            centroids = [X[labels == cluster].mean(axis=0) for cluster in clusters]
            centroid_distances = pairwise_distances(centroids)
            np.fill_diagonal(centroid_distances, np.nan)
            separation = np.nanmean(centroid_distances)
            
            return np.mean(cohesion), separation
        
        cohesion, separation = compute_cohesion_separation(self.X, clusters)
        metrics["Cohesion"] = cohesion
        metrics["Separation"] = separation
        
        # Average cluster size
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        metrics["Average Cluster Size"] = np.mean(cluster_counts)
        
        # Print metrics
        print(f"--- {model_name} Clustering Evaluation ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
        
        return metrics


    def visualize_clusters_2d(self, features_pca, clusters, title="Cluster Visualization", cluster_folder=None):
        """Visualizes clusters in 2D using PCA and saves the output in the specified folder."""
        plt.figure(figsize=(10, 7))
        plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap="viridis", s=50, alpha=0.7)
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster")
        plt.grid(True)

        # Define output path
        if cluster_folder:
            os.makedirs(cluster_folder, exist_ok=True)  # Ensure the folder exists
            output_path = os.path.join(cluster_folder, f"{title.replace(' ', '_')}.png")
        else:
            output_path = os.path.join("results/clustering/partitional", f"{title.replace(' ', '_')}.png")
        
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


    def cluster_size_distribution(self, clusters, cluster_folder=None):
        """
        Visualizes the distribution of cluster sizes and optionally saves it in a specific folder.
        
        Parameters:
            clusters (array-like): Cluster labels for the data points.
            cluster_folder (str, optional): Folder to save the visualization. Defaults to None.
        """
        unique, counts = np.unique(clusters, return_counts=True)
        plt.figure(figsize=(8, 5))
        plt.bar(unique, counts, color='skyblue')
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Points")
        
        if cluster_folder:
            os.makedirs(cluster_folder, exist_ok=True)
            output_path = os.path.join(cluster_folder, "Cluster_Size_Distribution.png")
        else:
            output_path = os.path.join("results/clustering", "Cluster_Size_Distribution.png")
        
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

    def cluster_characteristics(self, df, clusters, cluster_class):
        """
        Generates descriptive statistics and visualizes characteristics by cluster.
        Ensures the DataFrame has the correct structure and includes feature names.
        """
        # Ensure df is a DataFrame with the correct number of columns
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=self.feature_names)

        # Add cluster labels
        df['Cluster'] = clusters

        # Print descriptive statistics by cluster
        print("--- Cluster Statistics ---")
        print(df.groupby('Cluster').describe())

        # Visualize each feature by cluster
        for feature in self.feature_names:
            formatted_feature = feature.replace(" ", "_").replace("/", "_")
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='Cluster', y=feature, data=df)
            plt.title(f"{feature} by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel(feature)
            plt.grid(True)
            output_path = os.path.join(f"results/clustering/{cluster_class}/{formatted_feature}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")

    def save_results(self, results, filename, cluster_folder):
        """Save fine-tuning results to a CSV file within the cluster-specific folder."""
        os.makedirs(cluster_folder, exist_ok=True)  # Crear carpeta si no existe
        output_path = os.path.join(cluster_folder, filename)
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_path, index=False)
        print(f"Saved results to: {output_path}")

    def save_visualization(self, plt, filename, cluster_folder):
        """Save visualizations to the appropriate cluster-specific folder."""
        os.makedirs(cluster_folder, exist_ok=True)  # Crear carpeta si no existe
        output_path = os.path.join(cluster_folder, filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


class PartitionalClustering(UnsupervisedAlgorithms):
    def __init__(self, X, y=None, X_validation=None, y_validation=None, feature_names=None):
        super().__init__(X, y, X_validation, y_validation, feature_names)

    def fine_tune(self, max_clusters=15):
        """Fine-tuning KMeans by varying the number of clusters."""
        results = []
        for n_clusters in range(2, max_clusters + 1):
            cluster_folder = f"results/clustering/partitional/{n_clusters}"
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.X)
            metrics = self.evaluate_clustering(clusters, "K-Means")
            results.append({'Model': 'K-Means', 'n_clusters': n_clusters, **metrics})

        # Guardar resultados completos y obtener top 3
        df_results = pd.DataFrame(results)
        df_results['Score'] = df_results['Silhouette Score'] - df_results['Davies-Bouldin Index'] + df_results['Calinski-Harabasz Score']
        top_3 = df_results.nlargest(3, 'Score')
        self.save_results(top_3, "KMeans_FineTuning.csv", "results/clustering/partitional")
        print(top_3)
        return top_3

    def cluster_size_distribution(self, clusters, cluster_folder):
        """Visualizes the distribution of cluster sizes and saves the plot."""
        unique, counts = np.unique(clusters, return_counts=True)
        plt.figure(figsize=(8, 5))
        plt.bar(unique, counts, color='skyblue')
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Points")
        self.save_visualization(plt, "Cluster_Size_Distribution.png", cluster_folder)

    def analyze_top_models(self, top_models, features_pca, original_df):
        """Visualizes and analyzes the top 3 models."""
        for _, model_info in top_models.iterrows():
            n_clusters = model_info['n_clusters']
            cluster_folder = f"results/clustering/partitional/{n_clusters}"
            print(f"Analyzing K-Means with n_clusters={n_clusters}")

            # Create the model and predict clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.X)

            # Visualizations and analysis
            self.visualize_clusters_2d(features_pca, clusters, title=f"KMeans_{n_clusters}_Clusters", cluster_folder=cluster_folder)
            self.cluster_size_distribution(clusters, cluster_folder=cluster_folder)

            # Use the original DataFrame for cluster characteristics
            self.cluster_characteristics(df=original_df, clusters=clusters, cluster_class="partitional")

            # Save model details
            self.save_results([model_info.to_dict()], filename=f"Model_Details.csv", cluster_folder=cluster_folder)

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
    
    def fine_tune(self, max_clusters=15, linkage_methods=['ward', 'complete', 'average', 'single']):
        """Fine-tuning Hierarchical Clustering by varying linkage methods and the number of clusters."""
        results = []
        features_pca = PCA(n_components=2).fit_transform(self.X)  # Asegurar que PCA esté calculado
        for method in linkage_methods:
            for n_clusters in range(2, max_clusters + 1):
                cluster_folder = f"results/clustering/hierarchical/{n_clusters}"
                #os.makedirs(cluster_folder, exist_ok=True)
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
                clusters = model.fit_predict(self.X)
                metrics = self.evaluate_clustering(clusters, "Hierarchical")
                results.append({'Model': 'Hierarchical', 'Linkage': method, 'n_clusters': n_clusters, **metrics})
                
        df_results = pd.DataFrame(results)
        df_results['Score'] = df_results['Silhouette Score'] - df_results['Davies-Bouldin Index'] + df_results['Calinski-Harabasz Score']
        top_3 = df_results.nlargest(3, 'Score')
        self.save_results(top_3, "Hierarchical_FineTuning.csv", cluster_folder="results/clustering/hierarchical")
   
        return top_3

    def analyze_top_models(self, top_models, features_pca, df):
        """Visualizes and analyzes the top 3 models."""
        for _, model_info in top_models.iterrows():
            n_clusters = model_info['n_clusters']
            method = model_info['Linkage']
            print(f"Analyzing Hierarchical with n_clusters={n_clusters} and linkage={method}")
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
            clusters = model.fit_predict(self.X)

            # Visualizations and analysis
            self.visualize_clusters_2d(features_pca, clusters, title=f"Hierarchical_{method}_{n_clusters}_Clusters")
            self.cluster_size_distribution(clusters)
            self.cluster_characteristics(df, clusters, cluster_class="hierarchical")

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

    def fine_tune(self, max_components=15, cov_types=['full', 'tied', 'diag', 'spherical']):
        """Fine-tuning GMM by varying the number of components and covariance type."""
        results = []
        for cov_type in cov_types:
            for n_components in range(2, max_components + 1):
                cluster_folder = f"results/clustering/probabilistic/{n_components}"
                gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
                clusters = gmm.fit_predict(self.X)
                metrics = self.evaluate_clustering(clusters, "Probabilistic")
                results.append({'Model': 'GMM', 'Covariance Type': cov_type, 'n_components': n_components, **metrics})

        # Guardar resultados completos y obtener top 3
        df_results = pd.DataFrame(results)
        df_results['Score'] = df_results['Silhouette Score'] - df_results['Davies-Bouldin Index'] + df_results['Calinski-Harabasz Score']
        top_3 = df_results.nlargest(3, 'Score')
        self.save_results(top_3, "GMM_FineTuning.csv", "results/clustering/probabilistic")
        return top_3

    def analyze_top_models(self, top_models, features_pca, df):
        """Visualizes and analyzes the top 3 models."""
        for _, model_info in top_models.iterrows():
            n_components = model_info['n_components']  # Corrected from 'n_clusters' to 'n_components'
            cov_type = model_info['Covariance Type']
            print(f"Analyzing GMM with n_components={n_components} and covariance_type={cov_type}")

            # Create the model and predict clusters
            gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
            clusters = gmm.fit_predict(self.X)

            # Visualizations and analysis
            cluster_folder = f"results/clustering/probabilistic/{n_components}"
            self.visualize_clusters_2d(features_pca, clusters, title=f"GMM_{cov_type}_{n_components}_Clusters", cluster_folder=cluster_folder)
            self.cluster_size_distribution(clusters, cluster_folder=cluster_folder)
            self.cluster_characteristics(df=df, clusters=clusters, cluster_class="probabilistic")  # Fixed

            # Save model details
            self.save_results([model_info.to_dict()], filename=f"Model_Details.csv", cluster_folder=cluster_folder)

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

    # Configuración de PCA para visualización
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # Partitional Clustering
    print("Fine-Tuning KMeans...")
    partitional_clustering = PartitionalClustering(features, feature_names=feature_names)
    kmeans_top_3 = partitional_clustering.fine_tune(max_clusters=12)
    partitional_clustering.analyze_top_models(kmeans_top_3, features_pca, df)

    # Hierarchical Clustering
    print("Fine-Tuning Hierarchical...")
    hierarchical_clustering = HierarchicalClustering(features, feature_names=feature_names)
    hierarchical_top_3 = hierarchical_clustering.fine_tune(max_clusters=12)
    hierarchical_clustering.analyze_top_models(hierarchical_top_3, features_pca, df)

    # Probabilistic Clustering
    print("Fine-Tuning GMM...")
    probabilistic_clustering = ProbabilisticClustering(features, feature_names=feature_names)
    gmm_top_3 = probabilistic_clustering.fine_tune(max_components=12)
    probabilistic_clustering.analyze_top_models(gmm_top_3, features_pca, df)
