# Chapter 12: Custom Models and Training with TensorFlow

This notebook serves as a summary and submission for **Chapter 12** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

This chapter transitions from the high-level `tf.keras` API to TensorFlow's lower-level Python API. The main objective is to provide the tools and knowledge necessary for when you need extra control over your models and training loops.

---

## 📚 Key Concepts Summarized

This notebook reproduces the code and theoretical explanations for the following key concepts:

### 1. TensorFlow Basics
* **Tensors vs. Variables:** Using TensorFlow like NumPy for tensor operations. It covers the critical difference between immutable `tf.Tensor` objects and mutable `tf.Variable` objects (which are used for model weights and must be modified with methods like `.assign()`).
* **Data Structures:** A brief overview of other structures like sparse tensors, ragged tensors, and string tensors.

### 2. Customizing Keras Components
The notebook demonstrates how to build and save custom components for when the built-in Keras options are not sufficient.
* **Custom Loss Functions:** Creating custom losses, first as a simple Python function (`huber_fn`) and then as a subclass of `keras.losses.Loss` to make it saveable with its hyperparameters (by implementing `get_config`).
* **Custom Activations, Initializers, Regularizers, and Constraints:** Building simple, stateless custom components as Python functions and stateful components by subclassing their respective base classes (e.g., `keras.regularizers.Regularizer`).
* **Custom Metrics:** Explaining the difference between losses (for training) and metrics (for evaluation). It covers **streaming (stateful) metrics** (like `Precision`) that must be updated across batches and shows how to build one by subclassing `keras.metrics.Metric`.
* **Custom Layers:** Building **stateless** layers (using `keras.layers.Lambda`) and **stateful** layers with weights (by subclassing `keras.layers.Layer` and implementing `build()` and `call()`).
* **Custom Models:** Creating complex architectures (like a `ResidualRegressor`) by subclassing `keras.Model` and defining the forward pass in the `call()` method.
* **Losses Based on Model Internals:** Demonstrates how to add regularization losses (e.g., a reconstruction loss) directly inside the model's `call()` method using `self.add_loss()`.

### 3. Autodiff and `tf.GradientTape`
* Explains how to compute gradients automatically using a `tf.GradientTape()` context.
* Reproduces code for finding gradients of simple functions, using persistent tapes, and stopping gradients with `tf.stop_gradient()`.

### 4. Custom Training Loops
* Details the process of writing a training loop from scratch, which provides maximum control.
* The code demonstrates the full loop: iterating over epochs and batches, opening a `GradientTape`, making a prediction (forward pass), computing the loss, finding the gradients (`tape.gradient()`), and applying them to the model's weights (`optimizer.apply_gradients()`).

### 5. TensorFlow Functions and Graphs
* Explains how to dramatically improve performance by converting Python functions into static computation graphs using the `@tf.function` decorator.
* Covers the concepts of **Tracing** (converting the function to a graph) and **AutoGraph** (converting Python control flow like `if` and `for` into TF graph operations).
* Lists the rules to follow to ensure a function is convertible.

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
