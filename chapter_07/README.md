# Chapter 7: Ensemble Learning and Random Forests

This notebook serves as a summary and submission for **Chapter 7** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

It reproduces all the code from the chapter and provides theoretical explanations and summaries for each concept, as required by the assignment.

---

## 📚 Chapter Summary: Key Concepts

This chapter explores **Ensemble Learning**, a technique that aggregates the predictions of a group of predictors (an *ensemble*) to achieve a better prediction than any single predictor. This is often called the "wisdom of the crowd".

The main ensemble methods covered in this notebook are:

* **Voting Classifiers:** Combines different classifiers and makes a prediction based on the majority vote (hard voting) or the average of predicted probabilities (soft voting). This works best when the classifiers are diverse.
* **Bagging (Bootstrap Aggregating) and Pasting:** Uses the same training algorithm for every predictor but trains them on different random subsets of the training set. Bagging samples *with* replacement; Pasting samples *without* replacement.
* **Random Forests:** An ensemble of Decision Trees, typically trained via the bagging method. It introduces extra randomness by only considering a random subset of features when splitting a node.
    * **Extra-Trees (Extremely Randomized Trees):** A variant of Random Forests that uses random thresholds for each feature, making them faster to train.
* **Boosting (Hypothesis Boosting):** An ensemble method that combines several weak learners into a strong learner by training them sequentially, with each new predictor correcting its predecessor.
    * **AdaBoost (Adaptive Boosting):** Pays more attention to misclassified training instances from the previous predictor.
    * **Gradient Boosting:** Fits each new predictor to the *residual errors* made by the previous predictor.
* **Stacking (Stacked Generalization):** An ensemble where, instead of using a simple voting function, a final model (a *blender* or *meta-learner*) is trained on the predictions of all the other predictors in the ensemble.

---

## 💻 Code & Concepts Reproduced

This notebook provides code examples and theoretical explanations for the following topics:

* **Voting Classifiers:** Building and training a `VotingClassifier` with both hard and soft voting on the `make_moons` dataset.
* **Bagging and Pasting:** Implementing a `BaggingClassifier` (setting `bootstrap=True` for bagging, `bootstrap=False` for pasting) and evaluating its performance.
* **Random Forests:** Training a `RandomForestClassifier` and comparing it to a `BaggingClassifier` of `DecisionTreeClassifier`s. Also includes an example of finding feature importance.
* **Boosting:**
    * Implementing and training an `AdaBoostClassifier`.
    * Implementing and training a `GradientBoostingRegressor` (GBRT), including a method to find the optimal number of trees using early stopping.
* **Stacking:** The notebook concludes by describing the Stacking method, which is explored further in the end-of-chapter exercises.

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
