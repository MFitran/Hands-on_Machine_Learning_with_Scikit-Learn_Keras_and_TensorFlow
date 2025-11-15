# Chapter 10: Introduction to Artificial Neural Networks with Keras

This notebook serves as a summary and submission for **Chapter 10** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

This chapter introduces **Artificial Neural Networks (ANNs)**, the core of Deep Learning. It covers the history from the biological inspiration of the Perceptron to building modern Multilayer Perceptrons (MLPs) using the Keras API.

---

## 📚 Key Concepts Summarized

This notebook reproduces the code and theoretical explanations for the following key concepts:

* **The Perceptron:** A simple ANN architecture based on a Threshold Logic Unit (TLU). It covers its training method (Hebb's rule) and its key limitation: the inability to solve the XOR problem.
* **Multilayer Perceptron (MLP):** The evolution of the Perceptron, where stacking layers (input, hidden, and output) allows the network to solve complex problems.
* **Backpropagation:** The crucial training algorithm for MLPs. It works by performing a forward pass to get predictions, measuring the error, performing a backward pass to compute the error gradients, and then applying a Gradient Descent step.
* **Activation Functions:** The non-linear functions (like **ReLU**, **sigmoid**, and **tanh**) required between layers. Without them, a deep network would be equivalent to a simple linear model.

---

## 💻 Practical Implementation with Keras

This notebook builds and trains several neural networks using `tensorflow.keras`.

### 1. Classification MLP with the Sequential API
* **Dataset:** Fashion MNIST.
* **Model:** A `Sequential` model is built, which is a simple stack of layers.
* **Architecture:**
    1.  `Flatten` (to convert 28x28 images to a 1D array).
    2.  `Dense` hidden layer (300 neurons, ReLU activation).
    3.  `Dense` hidden layer (100 neurons, ReLU activation).
    4.  `Dense` output layer (10 neurons, **softmax** activation for multiclass classification).
* **Loss:** `sparse_categorical_crossentropy` is used because the labels are single integers (not one-hot vectors).

### 2. Regression MLP with the Sequential API
* **Dataset:** California Housing.
* **Model:** A `Sequential` model for regression.
* **Architecture:**
    1.  `Dense` hidden layer (30 neurons, ReLU activation).
    2.  `Dense` output layer (1 neuron, **no activation function**).
* **Loss:** `mean_squared_error` is used for this regression task.

### 3. Advanced Keras APIs
The notebook also explains and demonstrates Keras's more flexible APIs:
* **Functional API:** Used to build complex network architectures, such as a **Wide & Deep network** with multiple inputs (a "wide" path and a "deep" path) and multiple outputs (a main output and an auxiliary output for regularization).
* **Subclassing API:** Used to create fully dynamic models by subclassing `keras.Model` and defining the forward pass in the `call()` method.

### 4. Model Management and Tuning
Finally, the notebook covers essential practical skills for training and optimizing models:
* **Saving & Restoring:** Using `model.save()` and `keras.models.load_model()` to save and load models.
* **Callbacks:** Using callbacks to manage the training process:
    * `ModelCheckpoint`: Saves the best-performing model during training.
    * `EarlyStopping`: Interrupts training when validation performance stops improving.
* **TensorBoard:** Using the `TensorBoard` callback to log training history and visualize learning curves.
* **Hyperparameter Tuning:** Demonstrates how to wrap a Keras model (using `KerasRegressor`) to make it compatible with Scikit-Learn's `RandomizedSearchCV` for finding the best combination of hyperparameters.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `tensorflow` (version 2.0 or later)
* `keras` (bundled with TensorFlow)
* `scikit-learn`
* `numpy`
* `matplotlib`
* `pandas`
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install tensorflow scikit-learn numpy matplotlib pandas jupyter
