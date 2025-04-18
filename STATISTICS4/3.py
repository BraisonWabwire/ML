
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

# Read the data from Excel file
# Read the data from Excel file
data = pd.read_excel('C:/Users/brais/OneDrive/Desktop/ML/STATISTICS4/3.xlsx')

# Select the first five variables (x1, x2, x3, x4, x5)
X = data[['x1', 'x2', 'x3', 'x4', 'x5']].values

# Define matching coefficient as a similarity measure
def matching_coefficient(x, y):
    matches = np.sum(x == y)
    total = len(x)
    return matches / total

# Convert matching coefficient to a distance metric (1 - similarity)
def matching_distance(X):
    return 1 - squareform(pdist(X, metric=matching_coefficient))

# Part 1: Agglomerative clustering with average linkage, 4 clusters
# Compute distance matrix using matching coefficient
dist_matrix = matching_distance(X)

# Perform clustering with updated parameter
clustering_avg = AgglomerativeClustering(n_clusters=4, metric='precomputed', linkage='average')
labels_avg = clustering_avg.fit_predict(dist_matrix)

# Add cluster labels to the dataframe
data['cluster_avg'] = labels_avg

# Find the largest cluster
cluster_sizes = data['cluster_avg'].value_counts()
largest_cluster = cluster_sizes.idxmax()
largest_cluster_size = cluster_sizes.max()
print(f"Largest cluster (average linkage, 4 clusters): Cluster {largest_cluster} with {largest_cluster_size} points")

# Count number of 1s in x1 for the largest cluster
num_ones_x1_avg = data[data['cluster_avg'] == largest_cluster]['x1'].sum()
print(f"Number of 1s in x1 for the largest cluster (average linkage, 4 clusters): {num_ones_x1_avg}")

# Part 2: Agglomerative clustering with complete linkage, 3 clusters
clustering_complete = AgglomerativeClustering(n_clusters=3, metric='precomputed', linkage='complete')
labels_complete = clustering_complete.fit_predict(dist_matrix)

# Add cluster labels to the dataframe
data['cluster_complete'] = labels_complete

# Find the largest cluster
cluster_sizes_complete = data['cluster_complete'].value_counts()
largest_cluster_complete = cluster_sizes_complete.idxmax()
largest_cluster_size_complete = cluster_sizes_complete.max()
print(f"\nLargest cluster (complete linkage, 3 clusters): Cluster {largest_cluster_complete} with {largest_cluster_size_complete} points")

# Count number of 1s in x1 for the largest cluster
num_ones_x1_complete = data[data['cluster_complete'] == largest_cluster_complete]['x1'].sum()
print(f"Number of 1s in x1 for the largest cluster (complete linkage, 3 clusters): {num_ones_x1_complete}")