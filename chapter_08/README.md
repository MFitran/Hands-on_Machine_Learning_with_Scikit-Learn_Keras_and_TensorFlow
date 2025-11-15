# Chapter 8: Dimensionality Reduction

This notebook serves as a summary and submission for **Chapter 8** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

It reproduces all the code from the chapter and provides theoretical explanations and summaries for each key concept, as required by the assignment.

---

## 📚 Chapter Summary: Key Concepts

This chapter explores **Dimensionality Reduction**, the process of reducing the number of features (dimensions) in a dataset, which is a common step for speeding up training, data visualization, and fighting the "curse of dimensionality."

The notebook covers two main approaches:

* **Projection:** This approach projects the data onto a lower-dimensional subspace. The most popular projection technique is **Principal Component Analysis (PCA)**.
* **Manifold Learning:** This approach relies on the *manifold assumption*—that most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. Techniques covered include **Kernel PCA (kPCA)** and **Locally Linear Embedding (LLE)**.

Key concepts reproduced and explained:

* **The Curse of Dimensionality:** A theoretical explanation of why high-dimensional data is sparse, making training slow and increasing the risk of overfitting.
* **Principal Component Analysis (PCA):** The most popular projection technique. It identifies the axes (Principal Components) that preserve the maximum amount of variance in the data.
* **Explained Variance Ratio:** A metric used to select the right number of dimensions, often by choosing enough components to preserve a set percentage (e.g., 95%) of the total variance.
* **PCA for Compression & Reconstruction:** Using PCA to compress data (like the MNIST dataset) and then using its `inverse_transform` method to reconstruct it (with some information loss).
* **Randomized & Incremental PCA:** Variants of PCA for very large datasets. **Randomized PCA** is a stochastic algorithm that finds an approximation of the PCs much faster. **Incremental PCA (IPCA)** splits the dataset into mini-batches, allowing it to be used for datasets that don't fit in memory.
* **Kernel PCA (kPCA):** A nonlinear dimensionality reduction technique that applies the "kernel trick" (seen in SVMs) to PCA, allowing it to unroll complex manifolds like the Swiss roll.
* **Locally Linear Embedding (LLE):** Another powerful manifold learning technique that identifies the $k$ nearest neighbors for each instance and reconstructs it as a linear function of those neighbors, preserving local relationships.
* **Other Techniques:** The notebook also briefly describes other methods like Multidimensional Scaling (MDS), Isomap, t-SNE, and Linear Discriminant Analysis (LDA).

---

## 💻 Code & Concepts Reproduced

This notebook provides code examples and theoretical explanations for the following topics:

* **PCA with SVD:** Manually finding the Principal Components of a 3D dataset using `numpy`'s `svd` function.
* **PCA with Scikit-Learn:** Using `sklearn.decomposition.PCA` to perform the same task more easily.
* **Finding Explained Variance:** Demonstrating how to use `explained_variance_ratio_` to find the optimal number of dimensions and how to set `n_components=0.95` to do it automatically.
* **PCA for Compression:** Applying PCA to the MNIST dataset, reducing it from 784 to 154 dimensions (preserving 95% variance), and plotting an original vs. reconstructed digit.
* **Incremental PCA:** Implementing `IncrementalPCA` on the MNIST dataset in mini-batches.
* **Kernel PCA:** Applying `KernelPCA` with an RBF kernel to the `make_swiss_roll` dataset to unroll it into 2D.
* **Hyperparameter Tuning for kPCA:** Using `GridSearchCV` in a `Pipeline` with `KernelPCA` and `LogisticRegression` to find the best kernel and hyperparameters for a subsequent classification task.
* **LLE:** Applying `LocallyLinearEmbedding` to the `make_swiss_roll` dataset.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn` (`sklearn`)
* `pandas` (used for `qcut` in the hyperparameter tuning section)
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install numpy matplotlib scikit-learn pandas jupyter
