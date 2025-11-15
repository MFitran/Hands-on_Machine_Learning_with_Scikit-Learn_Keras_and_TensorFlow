# Chapter 3: Classification

This notebook serves as a submission for **Chapter 3** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

The chapter introduces **classification**, one of the most common supervised learning tasks. This notebook reproduces all the code from the chapter, using the **MNIST dataset** to explore key classification concepts and provide theoretical explanations for each step.

---

## 📚 Key Concepts Covered

This notebook walks through the entire process of building and evaluating classifiers, covering the following topics:

* **MNIST Dataset:** Loading the "hello world" of machine learning, a set of 70,000 handwritten digits.
* **Binary Classifier:** Training a simple "5-detector" to distinguish between two classes (5s and not-5s) using an `SGDClassifier`.
* **Performance Measures:** A deep dive into why accuracy is often a poor metric for classifiers, especially with skewed datasets. We explore better alternatives:
    * **Confusion Matrix:** Understanding True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
    * **Precision, Recall, and F1-Score:** Calculating and interpreting these essential metrics.
    * **Precision/Recall Trade-off:** Understanding how adjusting a classifier's decision threshold allows you to trade precision for recall.
    * **The ROC Curve:** Plotting the True Positive Rate (Recall) vs. the False Positive Rate (FPR) and using the **Area Under the Curve (AUC)** to compare models.
* **Multiclass Classification:**
    * Moving from binary to classifying all 10 digits.
    * Discussing strategies like **One-versus-the-Rest (OvR)** and **One-versus-One (OvO)**.
    * Training `SVC` (uses OvO) and `SGDClassifier` (handles multiclass natively) on the full dataset.
* **Error Analysis:**
    * Analyzing the multiclass confusion matrix to identify which digits the classifier struggles with.
    * Visualizing examples of common errors (e.g., 3s misclassified as 5s) to gain insight.
* **Multilabel Classification:**
    * Building a `KNeighborsClassifier` that outputs multiple binary tags for each instance (e.g., identifying a digit as both "large" and "odd").
* **Multioutput Classification:**
    * A generalization of multilabel classification where each label can be multiclass.
    * An example of building an **image denoising system** is reproduced, where the model's output is one value (0-255) for each of the 784 pixels.

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
