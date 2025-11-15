# Chapter 1: The Machine Learning Landscape

This Jupyter Notebook serves as a summary and submission for **Chapter 1** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

The notebook provides a high-level overview of Machine Learning, focusing on its fundamental theoretical concepts. It also reproduces and explains the single code example from the chapter, which demonstrates a simple model-based learning approach.

---

## 📚 Key Concepts Summarized

This notebook covers the core definitions and concepts introduced in the chapter:

### 1. What is Machine Learning?
* The science (and art) of programming computers so they can **learn from data** without being explicitly programmed.
* **Tom Mitchell's Definition:** A program learns from experience **E** on task **T** with performance measure **P**, if its performance on **T** (measured by **P**) improves with experience **E**.

### 2. Why Use Machine Learning?
* To solve problems that are too complex for traditional, rule-based solutions (e.g., speech recognition).
* To replace long lists of hand-tuned rules, resulting in systems that are easier to maintain.
* To build systems that can adapt to new or fluctuating data.
* To gain insights from large and complex datasets (Data Mining).

### 3. Types of Machine Learning Systems
The notebook outlines the main categories of ML systems:

* **Based on Human Supervision:**
    * **Supervised Learning:** Learns from labeled data (e.g., Classification, Regression).
    * **Unsupervised Learning:** Learns from unlabeled data (e.g., Clustering, Anomaly Detection, Dimensionality Reduction).
    * **Semisupervised Learning:** Uses a mix of labeled and unlabeled data.
    * **Reinforcement Learning (RL):** An "agent" learns by performing actions and receiving rewards or penalties.
* **Based on Learning Method:**
    * **Batch Learning (Offline):** Trained all at once on all available data.
    * **Online Learning (Incremental):** Trained sequentially on data instances or small "mini-batches."
* **Based on Generalization Strategy:**
    * **Instance-Based:** Learns examples by heart and generalizes using a similarity measure (e.g., k-Nearest Neighbors).
    * **Model-Based:** Builds a predictive model from the data.

### 4. Main Challenges of ML
* **"Bad Data" Problems:**
    * Insufficient quantity of training data.
    * Nonrepresentative data (due to sampling noise or bias).
    * Poor-quality data (errors, outliers, noise).
    * Irrelevant features.
* **"Bad Algorithm" Problems:**
    * **Overfitting:** The model is too complex and learns the noise in the training data; it fails to generalize to new data.
    * **Underfitting:** The model is too simple and fails to capture the underlying structure of the data.

### 5. Testing and Validating
* **Training Set:** The data used to train the model.
* **Validation Set:** A subset of the training data used to compare models and tune hyperparameters.
* **Test Set:** Data held aside until the very end to estimate the final model's generalization error on unseen data.

---

## 💻 Code Example: Life Satisfaction vs. GDP

The notebook includes one practical, model-based learning example.

**Objective:** Predict a country's "Life satisfaction" based on its "GDP per capita."

**Steps Performed:**
1.  **Fetch Data:** Downloads OECD life satisfaction data (`oecd_bli_2015.csv`) and IMF GDP per capita data (`gdp_per_capita.csv`).
2.  **Prepare Data:** Merges and cleans the two datasets into a single Pandas DataFrame.
3.  **Visualize:** Creates a scatter plot to show the positive linear relationship between the two variables.
4.  **Train Model:** Selects and trains a **Linear Regression** model (`sklearn.linear_model.LinearRegression`) on the data.
5.  **Predict:** Uses the trained model's `.predict()` method to estimate the life satisfaction for a new country (Cyprus) based on its GDP.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
