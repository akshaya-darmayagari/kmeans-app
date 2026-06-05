import os
import pandas as pd
import joblib
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Load and save dataset (dropping target labels for unsupervised training)
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df.to_csv("data/wine_clustering.csv", index=False)

# 2. Scale features (mandatory for distance-based clustering algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Automatically Determine Optimal K using Silhouette Score
best_k = 2
best_score = -1

# We test K values from 2 to 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"Optimal Cluster Count Automatically Selected: K = {best_k} (Silhouette Score: {best_score:.4f})")

# 4. Train K-Means Model using the automatically selected best_k
model = KMeans(n_clusters=best_k, init="k-means++", random_state=42, n_init=10)
model.fit(X_scaled)

# 5. Save artifacts
joblib.dump(model, "models/kmeans_model.pkl", compress=3)
joblib.dump(scaler, "models/scaler.pkl", compress=3)

print("K-Means clustering model trained and saved successfully.")