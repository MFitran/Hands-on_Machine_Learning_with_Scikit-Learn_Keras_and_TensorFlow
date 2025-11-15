# Chapter 5: Support Vector Machines

This notebook serves as a submission for **Chapter 5** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

It reproduces the code samples from the chapter and provides theoretical explanations and summaries for each key concept, as required by the assignment.

---

## 📚 Chapter Summary: Key Concepts

This chapter provides a deep dive into **Support Vector Machines (SVMs)**, a powerful and versatile set of machine learning models capable of linear or nonlinear classification, regression, and outlier detection.

The key concepts covered in this notebook include:

* **Large Margin Classification:** The fundamental idea of SVMs, which is to fit the "widest possible street" between two classes to find the optimal decision boundary.
* **Support Vectors:** The specific training instances that are located on the edge of this "street." They are the only points that "support" the decision boundary.
* **Hard vs. Soft Margins:**
    * **Hard Margin:** Strictly imposes that all instances must be off the street. This only works for perfectly, linearly separable data and is sensitive to outliers.
    * **Soft Margin:** A more flexible approach that allows for some *margin violations* (instances on the street or on the wrong side). This is controlled by the `C` hyperparameter.
* **Nonlinear SVMs & The Kernel Trick:** To handle nonlinear data, SVMs can either add polynomial features or (more efficiently) use the **kernel trick**. This technique achieves the result of high-degree feature transformation without the computational cost. Common kernels include:
    * **Polynomial Kernel**
    * **Gaussian RBF Kernel**
* **SVM Regression (SVR):** Reverses the SVM objective. Instead of separating classes, SVR tries to fit as many instances as possible *on* the street while limiting margin violations. The street width is controlled by the `epsilon` (ε) hyperparameter.
* **Under the Hood (Theory):** The notebook also summarizes the mathematical concepts behind SVMs, including the Primal vs. Dual problem and how the kernel trick works.

---

## 💻 Code & Concepts Reproduced

This notebook provides code examples and theoretical explanations for the following topics:

### 1. Linear SVM Classification
* **Theory:** Explains the concepts of large margin, support vectors, and the difference between Hard and Soft Margin classification. It details the role of the `C` hyperparameter (high `C` = fewer violations/narrower street, low `C` = more violations/wider street).
* **Code:** Demonstrates how to train a `LinearSVC` model on the Iris dataset, using a `Pipeline` to include `StandardScaler` (as SVMs are sensitive to feature scales).

### 2. Nonlinear SVM Classification
This section explores two methods for handling non-linearly separable data:

* **Using Polynomial Features:**
    * **Theory:** Explains that adding polynomial features (e.g., using `PolynomialFeatures`) can make a dataset linearly separable.
    * **Code:** Implements a `Pipeline` that combines `PolynomialFeatures`, `StandardScaler`, and `LinearSVC` to classify the `make_moons` dataset.
* **Using the Kernel Trick:**
    * **Theory:** Introduces the kernel trick as a highly efficient way to achieve the results of high-degree polynomial feature mapping without the computational overhead. It also covers the Gaussian RBF kernel and its `gamma` (γ) hyperparameter, which acts as a regularizer (high `gamma` = overfitting, low `gamma` = underfitting).
    * **Code:**
        * `SVC(kernel="poly")`: Trains an SVM using a 3rd-degree polynomial kernel.
        * `SVC(kernel="rbf")`: Trains an SVM using the Gaussian RBF kernel.

### 3. SVM Regression (SVR)
* **Theory:** Explains how SVR inverts the SVM objective to perform regression. It aims to fit data *within* an `epsilon`-insensitive margin (the "street").
* **Code:**
    * `LinearSVR`: Demonstrates linear regression.
    * `SVR(kernel="poly")`: Shows how to use a kernelized SVR for nonlinear regression tasks.

### 4. Theoretical Deep-Dive
The notebook concludes with a summary of the mathematics "under the hood" of SVMs, explaining:
* The Decision Function
* The Primal Problem (Hard and Soft Margin objectives)
* The Dual Problem
* The mathematical insight behind Kernelized SVMs

---

##  exercises

The notebook also lists the 10 end-of-chapter exercises for reference.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn` (version 0.20 or later)
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install numpy matplotlib scikit-learn jupyter
