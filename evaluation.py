import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load compressed artifacts
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset
df = pd.read_csv("data/wine_clustering.csv")
X_scaled = scaler.transform(df)

# Calculate WCSS/Inertia for different K values (Elbow Method)
wcss = []
k_range = range(1, 11)
for k in k_range:
    temp_kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    temp_kmeans.fit(X_scaled)
    wcss.append(temp_kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o', linestyle='--', color='indigo')
plt.title("Elbow Method (Optimal K Selection)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia / WCSS")
plt.xticks(k_range)

# Predict cluster labels
labels = model.predict(X_scaled)

# Plotting the 2D cluster projection using Alcohol vs Color Intensity
plt.subplot(1, 2, 2)
scatter = plt.scatter(df["alcohol"], df["color_intensity"], c=labels, cmap="rainbow", edgecolors='k', alpha=0.8)
plt.title("2D Projection (Alcohol vs Color Intensity)")
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.legend(*scatter.legend_elements(), title="Clusters")

plt.tight_layout()
plt.show()