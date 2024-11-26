from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class MLAlgorithms:
    def __init__(self, X, feature_names=None):
        self.X = X
        # Verifica si feature_names no es None y contiene elementos
        self.feature_names = list(feature_names) if feature_names is not None else [f"Feature {i}" for i in range(X.shape[1])]

# Clase base para algoritmos no supervisados
class UnsupervisedAlgorithms(MLAlgorithms):
    def __init__(self, X, feature_names=None):
        super().__init__(X, feature_names)

    def evaluate_clustering(self, clusters, model_name):
        """Evalúa un modelo de clustering con múltiples métricas."""
        silhouette_avg = silhouette_score(self.X, clusters)
        db_score = davies_bouldin_score(self.X, clusters)
        ch_score = calinski_harabasz_score(self.X, clusters)
        print(f"--- {model_name} Clustering Evaluation ---")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Davies-Bouldin Index: {db_score:.3f}")
        print(f"Calinski-Harabasz Score: {ch_score:.3f}")
        return silhouette_avg, db_score, ch_score

    def visualize_clusters_2d(self, features_pca, clusters, title="Cluster Visualization"):
        """Visualiza los clusters en 2D usando PCA."""
        plt.figure(figsize=(10, 7))
        plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap="viridis", s=50, alpha=0.7)
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster")
        plt.grid(True)
        plt.show()

    def cluster_size_distribution(self, clusters):
        """Visualiza la distribución de tamaños de clusters."""
        unique, counts = np.unique(clusters, return_counts=True)
        plt.figure(figsize=(8, 5))
        plt.bar(unique, counts, color='skyblue')
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Points")
        plt.show()

    def feature_importance_analysis(self, clusters):
        """Calcula la importancia de características en la formación de clusters y muestra nombres."""
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
        plt.show()

    def cluster_characteristics(self, clusters, df):
        """Genera estadísticas descriptivas y visualiza características por cluster."""
        df['Cluster'] = clusters
        print("--- Cluster Statistics ---")
        print(df.groupby('Cluster').describe())

        for feature in self.feature_names:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='Cluster', y=feature, data=df)
            plt.title(f"{feature} by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel(feature)
            plt.grid(True)
            plt.show()


# Subclase para clustering particional
class PartitionalClustering(UnsupervisedAlgorithms):
    def __init__(self, X, feature_names=None):
        super().__init__(X, feature_names)

    def determine_optimal_clusters_elbow(self, max_clusters=15):
        """Calcula el WCSS para encontrar el número óptimo de clusters con el método del codo."""
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
        plt.show()

        deltas = [wcss[i] - wcss[i + 1] for i in range(len(wcss) - 1)]
        second_derivative = [deltas[i] - deltas[i + 1] for i in range(len(deltas) - 1)]
        optimal_clusters = second_derivative.index(max(second_derivative)) + 2
        print(f"Optimal number of clusters determined: {optimal_clusters}")
        return optimal_clusters

    def kmeans_clustering(self, n_clusters=3):
        """Ejecuta K-Means Clustering."""
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
            df (DataFrame): Original DataFrame, including `Artist Name` and `Track Name`.
            model: Clustering model (must have `cluster_centers_` for K-Means).

        Returns:
            dict: Details of the song closest to the centroid of each cluster.
        """
        df['Cluster'] = clusters
        closest_points = {}

        if hasattr(model, "cluster_centers_"):  # Check if the model has explicit centroids (e.g., K-Means)
            for cluster_id, centroid in enumerate(model.cluster_centers_):
                cluster_data = df[df['Cluster'] == cluster_id]
                distances = np.linalg.norm(cluster_data.drop(columns=['Cluster', 'Class', 'Artist Name', 'Track Name']).values - centroid, axis=1)
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
        Filtra las instancias de un cluster específico con opciones de condiciones adicionales.
        Incluye `Artist Name` y `Track Name` en la salida.

        Args:
            clusters (array-like): Etiquetas de cluster para cada instancia.
            df (DataFrame): DataFrame original de los datos.
            cluster_id (int): ID del cluster que se quiere analizar.
            filter_conditions (dict, optional): Condiciones adicionales en formato {columna: valor}.

        Returns:
            DataFrame: Instancias que cumplen con el filtro, incluyendo `Artist Name` y `Track Name`.
        """
        df['Cluster'] = clusters
        filtered_df = df[df['Cluster'] == cluster_id]

        # Aplicar condiciones adicionales de filtro
        if filter_conditions:
            for column, value in filter_conditions.items():
                filtered_df = filtered_df[filtered_df[column] == value]

        # Seleccionar solo columnas clave para mostrar en la salida
        output_columns = ['Artist Name', 'Track Name', 'Class', 'Cluster']
        filtered_output = filtered_df[output_columns]

        print(f"--- Instances in Cluster {cluster_id} ---")
        print(filtered_output)
        return filtered_output

    def calculate_class_percentage_per_cluster(self, clusters, df):
        """
        Calcula el porcentaje del total de cada clase que pertenece a cada cluster.

        Args:
            clusters (array-like): Etiquetas de cluster para cada instancia.
            df (DataFrame): DataFrame original de los datos, debe incluir `Class`.

        Returns:
            dict: Diccionario con los porcentajes por clase y cluster.
        """
        df['Cluster'] = clusters
        total_class_counts = df['Class'].value_counts()  # Total de instancias por clase en todo el dataset
        cluster_percentages = {}

        for cluster_id in np.unique(clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            class_counts = cluster_data['Class'].value_counts()  # Conteo por clase en el cluster
            cluster_percentages[cluster_id] = {}

            for class_label, count in class_counts.items():
                total_count_for_class = total_class_counts[class_label]
                percentage = (count / total_count_for_class) * 100
                cluster_percentages[cluster_id][class_label] = percentage

        # Mostrar los porcentajes calculados
        print("--- Class Percentages per Cluster ---")
        for cluster_id, percentages in cluster_percentages.items():
            print(f"Cluster {cluster_id}:")
            for class_label, percentage in percentages.items():
                print(f"  Class {class_label}: {percentage:.2f}% of total instances")
        
        return cluster_percentages


    def get_classes_and_counts_by_cluster(self, clusters, df):
        """
        Identifica las clases (`Class`), cuenta cuántas instancias hay de cada clase por cluster
        y muestra dos canciones representativas (`Track Name`) y artistas (`Artist Name`) por clase.

        Args:
            clusters (array-like): Etiquetas de cluster para cada instancia.
            df (DataFrame): DataFrame original de los datos, debe incluir `Class`, `Artist Name`, `Track Name`.

        Returns:
            dict: Clases únicas y sus conteos por cluster, incluyendo dos canciones y artistas representativos.
        """
        df['Cluster'] = clusters
        cluster_class_counts = {}

        for cluster_id in np.unique(clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            class_counts = cluster_data['Class'].value_counts().to_dict()
            cluster_class_counts[cluster_id] = {'Class Counts': class_counts, 'Representative': {}}

            print(f"Cluster {cluster_id}:")
            for class_label, count in class_counts.items():
                # Seleccionar dos representaciones de canciones y artistas
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


# Código principal
if __name__ == '__main__':
    # Cargar datos
    df = pd.read_csv('dataset/clusters_standard_data.csv')
    features = df.drop(columns=['Class', 'Artist Name', 'Track Name']).values
    feature_names = df.drop(columns=['Class', 'Artist Name', 'Track Name']).columns

    # Instanciar clustering particional
    partitional_clustering = PartitionalClustering(features, feature_names)
    optimal_clusters = partitional_clustering.determine_optimal_clusters_elbow()

    # Ejecutar K-Means
    kmeans, base_clusters = partitional_clustering.kmeans_clustering(n_clusters=optimal_clusters)

    # Visualizar distribución y características de clusters
    partitional_clustering.cluster_size_distribution(base_clusters)
    partitional_clustering.cluster_characteristics(base_clusters, df)

    # Analizar importancia de características
    partitional_clustering.feature_importance_analysis(base_clusters)

    # PCA para visualización
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # Visualizar clusters
    partitional_clustering.visualize_clusters_2d(features_pca, base_clusters, title="Base Clusters")

    # Obtener clases presentes en cada cluster
    cluster_class_counts = partitional_clustering.get_classes_and_counts_by_cluster(base_clusters, df)

    closest_songs = partitional_clustering.get_closest_to_centroid(base_clusters, df, kmeans)

    class_percentages = partitional_clustering.calculate_class_percentage_per_cluster(base_clusters, df)