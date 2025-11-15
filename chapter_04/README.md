# Chapter 4: Training Models

This Jupyter Notebook serves as a summary and submission for **Chapter 4** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

This chapter dives deep into the *how* of model training. Instead of treating models as black boxes, this notebook explores the underlying algorithms. It reproduces code and provides theoretical explanations for the core methods of training linear models for both regression and classification.

---

## 📚 Key Concepts Summarized

This notebook covers the following core concepts and techniques:

### 1. Linear Regression
The notebook first explores Linear Regression and two distinct methods for training it:
* **The Normal Equation:** A direct, "closed-form" mathematical equation that finds the optimal model parameters (theta) that minimize the cost function in a single step.
* **Gradient Descent (GD):** An iterative optimization algorithm that tweaks parameters gradually to find the minimum of a cost function.

### 2. Gradient Descent (GD) Variants
The notebook explores three "flavors" of Gradient Descent, each with different trade-offs in speed and convergence:
* **Batch GD:** Calculates gradients based on the *entire* training set at every step. It is accurate but can be very slow on large datasets.
* **Stochastic GD (SGD):** Calculates gradients based on a *single random instance* at every step. It is much faster but more erratic, eventually settling near the minimum (requires a *learning schedule*).
* **Mini-batch GD:** A compromise that computes gradients on small random subsets of the data (mini-batches).

A comparison table in the notebook outlines the pros and cons of each method based on dataset size (m) and number of features (n).

### 3. Polynomial Regression
This section demonstrates how to use a linear model to fit non-linear data. This is achieved by adding powers of each feature (e.g., $x^2$, $x^3$) as new features, then training a linear model on this extended set.

### 4. Learning Curves
A key tool for diagnosing model performance. Learning curves are plots of the model's error on the training and validation sets as a function of the training set size. They are used to detect:
* **Underfitting:** When both training and validation error are high and have plateaued close together. This indicates the model is too simple.
* **Overfitting:** When the training error is low, but there is a significant gap between it and the much-higher validation error. This indicates the model is too complex and not generalizing well.
* **The Bias/Variance Trade-off:** The notebook also summarizes this fundamental concept, where bias (error from wrong assumptions/underfitting) and variance (error from sensitivity to data/overfitting) are in constant opposition.

### 5. Regularized Linear Models
To combat overfitting, the notebook covers several regularization (constraint) techniques:
* **Ridge Regression ($\ell_2$ norm):** Adds a regularization term that forces the model to keep its weights as small as possible.
* **Lasso Regression ($\ell_1$ norm):** Adds a term that tends to eliminate the weights of the least important features, effectively performing automatic feature selection.
* **Elastic Net:** A middle ground that mixes both Ridge and Lasso regularization.
* **Early Stopping:** A different regularization method for iterative algorithms (like GD) that simply stops training as soon as the validation error reaches its minimum.

### 6. Logistic & Softmax Regression
Finally, the notebook explores how these linear models are adapted for classification tasks:
* **Logistic Regression (Binary):** Estimates the probability that an instance belongs to a class by feeding the linear model's output into a *logistic (sigmoid) function*.
* **Softmax Regression (Multinomial):** A generalization of Logistic Regression that supports multiple classes directly by using the *softmax function* to estimate probabilities for each class.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn`
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install numpy matplotlib scikit-learn jupyter
