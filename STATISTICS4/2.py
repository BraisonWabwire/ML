import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Prepare the data
data = pd.read_excel('C:/Users/brais/OneDrive/Desktop/ML/STATISTICS4/5.xlsx')

# Create a DataFrame
df = pd.DataFrame(data)

# Step 2: Perform K-Means clustering without standardizing the variables
kmeans = KMeans(n_clusters=3, random_state=42)

# Selecting the columns 'x1', 'x3', and 'x5' for clustering
X = df[['x1', 'x3', 'x5']]

# Fit the model
kmeans.fit(X)

# Get the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Add the cluster label to the original DataFrame
df['Cluster'] = labels

# Step 3: Find the largest cluster
cluster_sizes = df['Cluster'].value_counts()
largest_cluster_label = cluster_sizes.idxmax()
largest_cluster_size = cluster_sizes.max()
largest_cluster_center = centers[largest_cluster_label]

print("Without Standardization:")
print(f"Largest Cluster Size: {largest_cluster_size}")
print(f"Largest Cluster Center (x1, x3, x5): {largest_cluster_center.round(2)}")

# Step 4: Perform K-Means clustering with standardized variables (Z-scores)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a new KMeans instance for standardized data
kmeans_scaled = KMeans(n_clusters=3, random_state=42)

# Fit the model on standardized data
kmeans_scaled.fit(X_scaled)

# Get the cluster centers and labels for standardized data
centers_scaled = kmeans_scaled.cluster_centers_
labels_scaled = kmeans_scaled.labels_

# Add the cluster label to the original DataFrame for scaled data
df['Cluster (Scaled)'] = labels_scaled

# Step 5: Find the largest cluster for standardized data
cluster_sizes_scaled = pd.Series(labels_scaled).value_counts()
largest_cluster_label_scaled = cluster_sizes_scaled.idxmax()
largest_cluster_size_scaled = cluster_sizes_scaled.max()
largest_cluster_center_scaled = centers_scaled[largest_cluster_label_scaled]

print("\nWith Standardization (Z-scores):")
print(f"Largest Cluster Size: {largest_cluster_size_scaled}")
print(f"Largest Cluster Center (x1, x3, x5) Z-scores: {largest_cluster_center_scaled.round(2)}")