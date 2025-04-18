import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

# Creating the DataFrame from the provided data
data = {
    'x1': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    'x2': [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    'x3': [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    'x4': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    'x5': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Use the first five variables (x1, x2, x3, x4, x5) for clustering
X = df[['x1', 'x2', 'x3', 'x4', 'x5']].values  # Convert to numpy array

# Compute the binary distance matrix
distance_matrix = pairwise_distances(X, metric='jaccard')

# Function to perform clustering and find number of 1s in x1 of largest cluster
def cluster_and_analyze(distance_matrix, linkage_method, n_clusters, method='binary'):
    # Perform hierarchical clustering with the specified method
    Z = linkage(distance_matrix, method=linkage_method)

    # Form flat clusters
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Add clusters to dataframe
    df['Cluster'] = clusters

    # Find the largest cluster
    largest_cluster_label = df['Cluster'].value_counts().idxmax()
    largest_cluster = df[df['Cluster'] == largest_cluster_label]
    
    # Count number of 1s in x1 of the largest cluster
    ones_in_x1 = largest_cluster['x1'].sum()

    # Print clear output
    print(f"Results for {linkage_method.capitalize()} Linkage with {n_clusters} Clusters:")
    print(f"  - Largest Cluster Label: {largest_cluster_label}")
    print(f"  - Number of 1s in x1 of Largest Cluster: {ones_in_x1}\n")
    
    # Optional: plot dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(f"{linkage_method.capitalize()} Linkage Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Jaccard Distance")
    plt.show()

# Perform agglomerative clustering with "average" linkage and 4 clusters
cluster_and_analyze(distance_matrix, linkage_method='average', n_clusters=4)

# Perform agglomerative clustering with "complete" linkage and 3 clusters
cluster_and_analyze(distance_matrix, linkage_method='complete', n_clusters=3)
