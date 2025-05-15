# Importing main librarys according to requirements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# -------------------------------------
# References:
# - Scikit-Learn Documentation. (n.d.). KMeans Clustering. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# - Matplotlib 3D Scatter Example. https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
# - StackAbuse. (2023). K-Means Clustering with Python. https://stackabuse.com/k-means-clustering-with-scikit-learn-in-python/

# -------------------------------------
# STEP 1: Load the CSV Data
# File contains 3D data: X, Y, Z coordinates
data = pd.read_csv('DMVA3T1.csv', header=None)
# We rename the columns for clarity, default is none
data.columns = ['x', 'y', 'z']

# -------------------------------------
# STEP 2: Perform K-Means Clustering
# Using K=3 based on assignment instructions
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(data)  # Add cluster labels to DataFrame
centers = kmeans.cluster_centers_           # Extract centroid coordinates

# -------------------------------------
# STEP 3: Set Up 3D Plot Main
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# -------------------------------------
# STEP 4: Define Styles for Each Cluster
# Assign unique colors and markers
colors = ['blue', 'red', 'green']
markers = ['o', '^', 's']

# -------------------------------------
# STEP 5: Plot Data Points per Cluster
# Loops through each cluster to customize it
for cluster_id in range(3):
    cluster_data = data[data['cluster'] == cluster_id]
    ax.scatter(
        cluster_data['x'], cluster_data['y'], cluster_data['z'],
        c=colors[cluster_id],
        marker=markers[cluster_id],
        s=50,
        label=f'Cluster {cluster_id + 1}',
        alpha=0.8,
        edgecolors='white'
    )

# -------------------------------------
# STEP 6: Plot the Centroid Locations as X
ax.scatter(
    centers[:, 0], centers[:, 1], centers[:, 2],
    c='black',
    marker='X',
    s=300,
    label='Centroids',
    edgecolors='yellow'
)

# -------------------------------------
# STEP 7: Customize Axes and View
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.set_zlabel('Z-axis', fontsize=12)
ax.set_title('3D Scatter Plot with K-Means Clustering (k=3)', fontsize=16, pad=20)
ax.view_init(elev=30, azim=55) # elevation and Azim is to configure the view

# -------------------------------------
# STEP 8: Add legend and grid
ax.legend(loc='upper right')
ax.grid(True)

# -------------------------------------
# STEP 9: Save & Show Plot
# Save to PNG for reporting
plt.tight_layout()
plt.savefig('DMV302T1b.png', dpi=300, bbox_inches='tight')
plt.show()
