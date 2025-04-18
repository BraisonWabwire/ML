import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Read the data from Excel file
data = pd.read_excel('C:/Users/brais/OneDrive/Desktop/ML/STATISTICS4/7.xlsx')
# Select all four variables for clustering
X = data[['Sepal length', 'Sepal width', 'Petal length', 'Petal width']].values

# Function to perform k-means clustering and report largest cluster
def perform_kmeans(X, k, random_state=42):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    # Add cluster labels to the dataframe
    data['cluster'] = labels
    
    # Find the largest cluster
    cluster_sizes = data['cluster'].value_counts()
    largest_cluster = cluster_sizes.idxmax()
    largest_cluster_size = cluster_sizes.max()
    
    # Get cluster center values for the largest cluster
    cluster_centers = kmeans.cluster_centers_
    largest_cluster_center = cluster_centers[largest_cluster]
    
    # Round center values to 2 decimal places
    largest_cluster_center_rounded = np.round(largest_cluster_center, 2)
    
    return largest_cluster, largest_cluster_size, largest_cluster_center_rounded

# Part a: K-means clustering with k=4
largest_cluster_k4, size_k4, centers_k4 = perform_kmeans(X, k=4)
print(f"Part a: K-means clustering with k=4")
print(f"Largest cluster: Cluster {largest_cluster_k4} with size {size_k4}")
print(f"Cluster center values (Sepal length, Sepal width, Petal length, Petal width): "
      f"{centers_k4[0]}, {centers_k4[1]}, {centers_k4[2]}, {centers_k4[3]}")

# Part b: K-means clustering with k=3 and k=5
# k=3
largest_cluster_k3, size_k3, centers_k3 = perform_kmeans(X, k=3)
print(f"\nPart b: K-means clustering with k=3")
print(f"Largest cluster: Cluster {largest_cluster_k3} with size {size_k3}")
print(f"Cluster center values (Sepal length, Sepal width, Petal length, Petal width): "
      f"{centers_k3[0]}, {centers_k3[1]}, {centers_k3[2]}, {centers_k3[3]}")

# k=5
largest_cluster_k5, size_k5, centers_k5 = perform_kmeans(X, k=5)
print(f"\nPart b: K-means clustering with k=5")
print(f"Largest cluster: Cluster {largest_cluster_k5} with size {size_k5}")
print(f"Cluster center values (Sepal length, Sepal width, Petal length, Petal width): "
      f"{centers_k5[0]}, {centers_k5[1]}, {centers_k5[2]}, {centers_k5[3]}")