# Importing main librarys according to requirements
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
# References
# Matplotlib Docs: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
# StackAbuse. https://stackabuse.com/matplotlib-scatter-plot-with-distribution-plots-histograms-jointplot/ 

# ------------------------------
# STEP 1: Read the CSV data and store it into a variable
# ------------------------------
data = pd.read_csv('DMVA3T1.csv', header=None)
data.columns = ['x', 'y', 'z']

# ------------------------------
# STEP 2: Projection based on Student ID
# Because student ID ends with 7 so we use x-z plane
dimension1 = 'x'
dimension2 = 'z'

# ------------------------------
# STEP 3: Set up the main plot layout
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(4, 4)

# ------------------------------
# STEP 4: Making scatterplot main
ax_scatter = fig.add_subplot(gs[1:4, 0:3])
scatter = ax_scatter.scatter(
    data[dimension1], data[dimension2],
    c=data[dimension2], cmap='plasma', edgecolors='white', s=40, alpha=0.8
)
ax_scatter.set_xlabel(f'X Axis', fontsize=12)
ax_scatter.set_ylabel(f'Z Axis', fontsize=12)
ax_scatter.set_title(f'Scatter Plot (X - Z Plane)', fontsize=14, weight='bold')
ax_scatter.grid(True, linestyle='--', linewidth=0.5)

# ------------------------------
# STEP 5: Histogram of X / Top above scatter
ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
ax_histx.hist(data[dimension1], bins=30, color='#00BFFF', edgecolor='white', alpha=0.8)
ax_histx.set_ylabel('Frequency', fontsize=10)
ax_histx.set_title(f'Histogram of X', fontsize=12)
ax_histx.grid(True, linestyle=':', linewidth=0.5)
plt.setp(ax_histx.get_xticklabels(), visible=False)

# ------------------------------
# STEP 6: Histogram of Z / Right side
ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
ax_histy.hist(data[dimension2], bins=30, orientation='horizontal', color='#9f7bff', edgecolor='white', alpha=0.8)
ax_histy.set_xlabel('Frequency', fontsize=10)
ax_histy.set_title(f'Histogram of Z', fontsize=12)
ax_histy.grid(True, linestyle=':', linewidth=0.5)
plt.setp(ax_histy.get_yticklabels(), visible=False)

# ------------------------------
# STEP 7: Final layout adjustments and save the png
plt.tight_layout()
plt.savefig('DMV302T1a.png', dpi=300, bbox_inches='tight')
plt.show()  