import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from time import time

#dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, n_classes=6, random_state=42)

#  %70 Train, %30 Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', s=10)
plt.title("Original Training Data")

# K-means
kmeans = KMeans(n_clusters=20, random_state=42)  # 20 küme
kmeans.fit(X_train)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.subplot(2, 2, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=labels, cmap='tab10', s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
plt.title("Clustered Training Data")
plt.legend()

def calculate_density(cluster_idx, centroids, neighbors=5):
    """Her kümenin yoğunluğunu hesaplar."""
    nn = NearestNeighbors(n_neighbors=neighbors)
    nn.fit(centroids)
    distances, _ = nn.kneighbors([centroids[cluster_idx]])
    return 1 / np.sum(distances)

remaining_clusters = list(range(20))
densities = [calculate_density(i, centroids) for i in remaining_clusters]
num_clusters_to_keep = 10

while len(remaining_clusters) > num_clusters_to_keep:
    max_density_idx = np.argmax([densities[i] for i in remaining_clusters])
    remaining_clusters.remove(remaining_clusters[max_density_idx])

single_stage_indices = [i for i, label in enumerate(labels) if label in remaining_clusters]
X_train_ss = X_train[single_stage_indices]
y_train_ss = y_train[single_stage_indices]

plt.subplot(2, 2, 3)
plt.scatter(X_train_ss[:, 0], X_train_ss[:, 1], c=y_train_ss, cmap='tab10', s=10)
plt.title("Single-Stage Sampled Data")

np.random.seed(42)
double_stage_indices = np.random.choice(len(X_train_ss), size=len(X_train_ss) // 2, replace=False)
X_train_ds = X_train_ss[double_stage_indices]
y_train_ds = y_train_ss[double_stage_indices]

plt.subplot(2, 2, 4)
plt.scatter(X_train_ds[:, 0], X_train_ds[:, 1], c=y_train_ds, cmap='tab10', s=10)
plt.title("Double-Stage Sampled Data")
plt.tight_layout()
plt.show()

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

datasets = {
    "Original Data": (X_train, y_train),
    "Single-Stage Clustering": (X_train_ss, y_train_ss),
    "Double-Stage Clustering": (X_train_ds, y_train_ds),
}

results = {}

for name, (X, y) in datasets.items():
    start_time = time()
    mlp.fit(X, y)
    training_time = time() - start_time

    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {"Accuracy": accuracy, "Training Time (ms)": training_time * 1000}

print("\nEvaluation Results:")
for name, metrics in results.items():
    print(f"{name}: Mean Testing Accuracy: {metrics['Accuracy']:.3f}, Training Time: {metrics['Training Time (ms)']:.3f} ms")
