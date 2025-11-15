# Chapter 9: Unsupervised Learning Techniques

This notebook serves as a summary and submission for **Chapter 9** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

It reproduces all the code from the chapter and provides theoretical explanations and summaries for each concept, as required by the assignment.

---

## 📚 Chapter Summary: Key Concepts

This chapter explores several key **Unsupervised Learning** techniques, where models learn from data that has no labels. The notebook focuses on two primary areas: clustering and density estimation.

### 1. Clustering
The notebook covers the task of grouping similar instances together into clusters and explores two main algorithms:

#### K-Means
* **Algorithm:** A simple and efficient algorithm that finds a predefined number of clusters ($k$) by identifying cluster centers (centroids) and assigning instances to the nearest one.
* **Optimal *k*:** Explains and demonstrates two techniques to find the optimal number of clusters:
    * **Elbow Method:** Plotting inertia as a function of $k$ to find the "elbow" where the drop in inertia slows down.
    * **Silhouette Score:** A more precise metric that measures how well-separated clusters are by calculating the silhouette coefficient for each instance.
* **Applications (with code):**
    * **Image Segmentation:** Using K-Means to cluster pixel colors and reduce the number of colors in an image.
    * **Preprocessing:** Using clustering as a dimensionality reduction tool before a supervised learning algorithm (e.g., Logistic Regression on the digits dataset), which improves performance.
    * **Semi-Supervised Learning:** Demonstrates how to use clustering (on the digits dataset) to find representative instances, manually label only those few, and then propagate the labels to the rest of the cluster to train a model.

#### DBSCAN
* **Algorithm:** A density-based algorithm (Density-Based Spatial Clustering of Applications with Noise) that defines clusters as continuous regions of high density.
* **Advantages:** It is powerful because it can find clusters of arbitrary shapes and is robust to outliers (which it identifies as anomalies).
* **Code:** Implements DBSCAN on the `make_moons` dataset and shows how to use a `KNeighborsClassifier` to predict which cluster a new instance would belong to.

### 2. Gaussian Mixture Models (GMMs)
This section covers GMMs, which are probabilistic models that assume the data is generated from a mixture of several Gaussian distributions.

* **Algorithm:** Explains how GMMs are trained with the **Expectation-Maximization (EM)** algorithm for soft clustering (assigning probabilities to clusters).
* **Anomaly Detection:** Demonstrates how to use a trained GMM's density estimation capabilities (`score_samples()`) to identify instances in low-density regions as anomalies.
* **Finding Optimal *k*:** Since inertia and silhouette scores are unreliable for GMMs, this notebook shows how to use theoretical information criteria to find the best number of clusters:
    * **Bayesian Information Criterion (BIC)**
    * **Akaike Information Criterion (AIC)**
* **Bayesian Gaussian Mixture Models:** Implements the `BayesianGaussianMixture` class, which can automatically find the optimal number of clusters by setting unnecessary cluster weights to zero.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn` (`sklearn`)
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install numpy matplotlib scikit-learn jupyter
