import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read the data from Excel file
data = pd.read_excel('C:/Users/brais/OneDrive/Desktop/ML/STATISTICS4/5.xlsx')
# Select the variables x1, x3, and x5 for clustering
X = data[['x1', 'x3', 'x5']].values

# Part a: K-means clustering without standardization (3 clusters)
# Set random seed for reproducibility
kmeans = KMeans(n_clusters=3, random_state=42)
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

print(f"Part a: K-means clustering without standardization (3 clusters)")
print(f"Largest cluster: Cluster {largest_cluster} with size {largest_cluster_size}")
print(f"Cluster center values for x1, x3, x5: {largest_cluster_center_rounded[0]}, {largest_cluster_center_rounded[1]}, {largest_cluster_center_rounded[2]}")

# Part b: K-means clustering with standardization (3 clusters)
# Standardize the variables to z-scores
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Perform k-means clustering on standardized data
kmeans_std = KMeans(n_clusters=3, random_state=42)
labels_std = kmeans_std.fit_predict(X_standardized)

# Add standardized cluster labels to the dataframe
data['cluster_std'] = labels_std

# Find the largest cluster
cluster_sizes_std = data['cluster_std'].value_counts()
largest_cluster_std = cluster_sizes_std.idxmax()
largest_cluster_size_std = cluster_sizes_std.max()

# Get cluster center values for the largest cluster (in standardized units)
cluster_centers_std = kmeans_std.cluster_centers_
largest_cluster_center_std = cluster_centers_std[largest_cluster_std]

# Round center values to 2 decimal places
largest_cluster_center_std_rounded = np.round(largest_cluster_center_std, 2)

print(f"\nPart b: K-means clustering with standardization (3 clusters)")
print(f"Largest cluster: Cluster {largest_cluster_std} with size {largest_cluster_size_std}")
print(f"Cluster center values for x1, x3, x5 (z-scores): {largest_cluster_center_std_rounded[0]}, {largest_cluster_center_std_rounded[1]}, {largest_cluster_center_std_rounded[2]}")