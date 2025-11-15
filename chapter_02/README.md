# Chapter 2: End-to-End Machine Learning Project

This Jupyter Notebook serves as a submission for **Chapter 2** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

The notebook reproduces the code and theoretical explanations for walking through an end-to-end machine learning project. The objective is to build a model to predict median housing prices in California districts using census data.

---

## 📈 Project Workflow

This project follows the typical steps of a data science project, from problem definition to final model evaluation.

### 1. The Big Picture
* **Problem Framing:** The task is defined as a **supervised learning**, **multiple regression** problem.
* **Performance Measure:** **Root Mean Square Error (RMSE)** is selected as the primary metric to evaluate the model's performance.

### 2. Get the Data
* The notebook includes functions to fetch and load the California housing data (`housing.tgz`) from its source repository.

### 3. Data Discovery and Visualization
* The data is inspected using `.info()` and `.describe()` to identify its structure, features, and missing values (e.g., in `total_bedrooms`).
* Histograms are plotted for all numerical attributes to observe their distributions, identifying capped values (like `median_house_value`) and tail-heavy distributions.
* A **stratified test set** is created based on the `median_income` attribute to ensure the test set is representative of the whole population and to avoid data snooping bias.
* Geographical data is visualized using scatter plots, with `alpha` settings to show density.
* Price data is overlaid on the map (using `c` for color and `s` for size) to visualize where the most expensive districts are located.
* Correlation matrices and scatter plots are used to find promising correlations, especially between `median_house_value` and `median_income`.

### 4. Data Preparation
A full data transformation pipeline is built using Scikit-Learn:
* **Missing Values:** `SimpleImputer` is used to fill in missing `total_bedrooms` values with the median.
* **Categorical Data:** The text-based `ocean_proximity` feature is converted to numbers using `OneHotEncoder`.
* **Feature Engineering:** A custom transformer (`CombinedAttributesAdder`) is built to create new, more predictive features like `rooms_per_household` and `bedrooms_per_room`.
* **Feature Scaling:** All numerical attributes are standardized using `StandardScaler`.
* **Pipeline:** A `Pipeline` is used for all numerical transformations, and a `ColumnTransformer` combines the numerical pipeline and the `OneHotEncoder` into a single transformer that prepares the entire dataset.

### 5. Model Selection and Training
Several models are trained and evaluated using **K-fold cross-validation**:
* **Linear Regression:** Serves as a baseline model. It is found to be **underfitting** the data (RMSE: ~68,628).
* **Decision Tree Regressor:** This model is trained and found to be **overfitting** the data badly (RMSE: 0.0 on the training set, but ~71,415 on validation).
* **Random Forest Regressor:** This ensemble model performs the best (mean RMSE: ~50,129), making it the most promising candidate.

### 6. Model Fine-Tuning
The `RandomForestRegressor` is fine-tuned to find the best hyperparameters:
* **Grid Search:** `GridSearchCV` is used to explore multiple combinations of hyperparameters.
* **Randomized Search:** `RandomizedSearchCV` is used for a more efficient search over a wider range of hyperparameter values.
* **Feature Importance:** The best model's `feature_importances_` are analyzed to see which attributes contribute most to the predictions.

### 7. Final Evaluation
* The full data pipeline is run on the (previously untouched) **test set**.
* The final model achieves an RMSE of **$48,051** on the test data.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
