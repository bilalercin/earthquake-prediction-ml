import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# MNIST
mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

X_sample, _, y_sample, _ = train_test_split(data, labels, train_size=1000, stratify=labels)

# data normalize
X_sample = X_sample / 255.0

n_components_list = [10, 25, 500]
reconstructed_images_list = []

for n_components in n_components_list:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_sample)
    reconstructed_images = pca.inverse_transform(X_pca)
    reconstructed_images_list.append(reconstructed_images)


def plot_comparison(original, reconstructed, ax, title):
    ax.imshow(original.reshape(28, 28), cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    ax.imshow(reconstructed.reshape(28, 28), cmap='gray')
    ax.set_title(title)
    ax.axis('off')


fig, axs = plt.subplots(10, 4, figsize=(16, 20))
for digit in range(10):

    indices = np.where(y_sample == str(digit))[0][:3]
    for i, index in enumerate(indices):

        plot_comparison(X_sample.iloc[index].values, X_sample.iloc[index].values, axs[digit, 0], "Original Image")

        for comp_index, n_components in enumerate(n_components_list):
            reconstructed_image = reconstructed_images_list[comp_index][index]
            epoch_titles = [f"Components: {n_components}"]
            plot_comparison(X_sample.iloc[index].values, reconstructed_image, axs[digit, comp_index + 1],
                            epoch_titles[0])

plt.tight_layout()
plt.show()
