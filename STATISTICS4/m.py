import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Excel file
file_path = r'C:\Users\brais\OneDrive\Desktop\ML\STATISTICS4\Ch14_Q1_V10_Data_File.xlsx'
df = pd.read_excel(file_path)

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Function to plot dendrogram and print number of clusters at distance threshold
def plot_dendrogram(X, method, threshold=5):
    Z = linkage(X, method=method, metric='euclidean')
    plt.figure(figsize=(12, 5))
    plt.title(f"Agglomerative Clustering Dendrogram ({method.capitalize()} Linkage)")
    dendrogram(Z)
    plt.axhline(y=threshold, c='red', linestyle='--')
    plt.xlabel("Data Index")
    plt.ylabel("Distance")
    plt.show()

    clusters = fcluster(Z, t=threshold, criterion='distance')
    num_clusters = len(set(clusters))
    print(f"\n{method.capitalize()} linkage: Number of clusters at distance threshold {threshold} = {num_clusters}\n")

# Step 3: Perform clustering for each method

# Single Linkage
plot_dendrogram(X_scaled, method='single')

# Complete Linkage
plot_dendrogram(X_scaled, method='complete')

# Ward Linkage
plot_dendrogram(X_scaled, method='ward')
