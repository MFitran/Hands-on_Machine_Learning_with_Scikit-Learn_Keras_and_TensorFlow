# Chapter 6: Decision Trees

This notebook serves as a summary and submission for **Chapter 6** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

It reproduces all the code from the chapter and provides theoretical explanations and summaries for each key concept, as required by the assignment.

---

## 📚 Key Concepts Summarized

This chapter introduces **Decision Trees**, a versatile and powerful machine learning algorithm capable of performing both classification and regression tasks. Unlike "black box" models like SVMs, Decision Trees are intuitive, and their decisions are easy to interpret, making them "white box" models.

Key concepts covered include:

* **Training and Visualization:** How to train a `DecisionTreeClassifier` and visualize the resulting tree structure using `export_graphviz`.
* **Making Predictions:** A prediction is made by tracing a path from the root node down to a leaf node. The chapter explains key node attributes like `samples`, `value`, and `gini`.
* **Gini Impurity:** The cost function used by the CART algorithm to measure the "purity" of a node. A node is pure (Gini=0) if all instances it applies to belong to the same class.
* **Estimating Probabilities:** A Decision Tree estimates the probability of a class by returning the ratio of training instances of that class in the specific leaf node the instance falls into.
* **The CART Algorithm:** Scikit-Learn uses the Classification and Regression Tree (CART) algorithm, a *greedy algorithm* that searches for the feature and threshold combination that produces the purest subsets (by minimizing the cost function).
* **Regularization:** Decision Trees are prone to overfitting if left unconstrained. Regularization is achieved by setting hyperparameters like `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_leaf_nodes`.
* **Regression:** Decision Trees can also perform regression (`DecisionTreeRegressor`). Instead of predicting a class, each leaf node predicts a value (the average target value of the instances in that node). The CART algorithm works similarly but aims to minimize Mean Squared Error (MSE) instead of impurity.
* **Instability:** A key limitation of Decision Trees is their high sensitivity to small variations in the training data and to data rotation.

---

## 💻 Code & Concepts Reproduced

This notebook provides code examples and theoretical explanations for the following topics:

### 1. Training and Visualizing a Classifier
* **Code:** Trains a `DecisionTreeClassifier` on the Iris dataset, limited to `max_depth=2`.
* **Visualization:** Uses `export_graphviz` to create a `.dot` file and visualizes the tree structure.
* **Theory:** Explains how to make predictions by following the tree's branches and how to read the `samples`, `value`, and `gini` attributes at each node.

### 2. Estimating Class Probabilities
* **Theory:** Explains that probabilities are calculated from the ratio of class instances in the leaf node where a sample lands.
* **Code:** Uses `tree_clf.predict_proba()` and `tree_clf.predict()` to demonstrate this on a sample instance.

### 3. The CART Algorithm
* **Theory:** Summarizes the Classification and Regression Tree (CART) algorithm as a *greedy* process that recursively splits the data by finding the feature/threshold pair that minimizes a cost function (Gini impurity for classification, MSE for regression).

### 4. Regularization
* **Theory:** Discusses why Decision Trees tend to overfit and how to regularize them using hyperparameters like `max_depth`, `min_samples_split`, and `min_samples_leaf`.
* **Code:** Trains two trees on the `make_moons` dataset—one with no restrictions and one regularized with `min_samples_leaf=4`—to visually demonstrate the effect of regularization on overfitting.

### 5. Regression
* **Theory:** Explains how `DecisionTreeRegressor` works by predicting the average target value of the instances in a leaf node and minimizing MSE during training.
* **Code:** Trains regressors on a noisy quadratic dataset, first comparing `max_depth=2` vs. `max_depth=3`, and then comparing an unregularized tree vs. one regularized with `min_samples_leaf=10`.

### 6. Instability
* **Theory:** Highlights the key limitations of Decision Trees: their sensitivity to small data variations and, in particular, to dataset rotation.
* **Code:** Provides a clear visual example of how a simple 45° rotation of a dataset leads to a completely different and more complex decision boundary.

---

##  exercises

The notebook concludes by listing the 8 end-of-chapter exercises for reference.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn` (version 0.20 or later)
* `graphviz` (for visualizing the tree, or you can use the command-line `dot` tool)
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install numpy matplotlib scikit-learn graphviz jupyter
