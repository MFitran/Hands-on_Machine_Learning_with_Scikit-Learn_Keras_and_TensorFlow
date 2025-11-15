# Chapter 6: Decision Trees

This notebook serves as a summary and submission for **Chapter 6** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

It reproduces all the code from the chapter and provides theoretical explanations and summaries for each key concept, as required by the assignment.

---

## 📚 Key Concepts Summarized

This chapter introduces **Decision Trees**, a versatile and powerful machine learning algorithm capable of performing both classification and regression tasks. Unlike "black box" models, Decision Trees are intuitive, and their decisions are easy to interpret, making them "white box" models.

Key concepts covered include:

* **Training and Visualization:** How to train a `DecisionTreeClassifier` and visualize the resulting tree structure using `export_graphviz`.
* **Making Predictions:** We trace the path from the root node down to a leaf node to make a prediction. The chapter explains key attributes like `samples`, `value`, and `gini`.
* **Gini Impurity:** This is the cost function used by the CART algorithm. It measures the "purity" of a node. A node is pure (Gini=0) if all instances it applies to belong to the same class.
* **The CART Algorithm:** Scikit-Learn uses the Classification and Regression Tree (CART) algorithm, which is a *greedy algorithm* that searches for the feature and threshold that produce the purest subsets.
* **Regularization:** Decision Trees are prone to overfitting if left unconstrained. Regularization is achieved by setting hyperparameters like `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_leaf_nodes`.
* **Regression:** Decision Trees can also perform regression (`DecisionTreeRegressor`). Instead of predicting a class, each node predicts a value (the average target value of the training instances). The CART algorithm works similarly but aims to minimize the Mean Squared Error (MSE).
* **Instability:** The chapter concludes by highlighting a key limitation of Decision Trees: they are very sensitive to small variations in the training data and to data rotation.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn`
* `jupyter` (to run the notebook environment)
* `graphviz` (to visualize the tree structures directly in the notebook)

You can install these dependencies using `pip`:
```bash
pip install numpy matplotlib scikit-learn jupyter graphviz
