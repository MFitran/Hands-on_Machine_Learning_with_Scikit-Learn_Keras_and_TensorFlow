# Chapter 16: Deploying TensorFlow Models

This Jupyter Notebook (`notebook_16.ipynb`) contains the code and theoretical summaries from **Chapter 16 of "TensorFlow in Action" by Thushan Ganegedara**.

This notebook provides a practical overview of the model deployment lifecycle in TensorFlow. It covers saving models in different formats and then deploying them to three distinct environments: a high-performance production server, an on-device mobile application, and a web browser.

---

##  summary-of-contents

### 1. Saving and Loading Models

Before a model can be deployed, it must be saved. This section demonstrates the two primary ways to save a Keras/TensorFlow model using an MNIST classifier as the example.

* **Keras `.h5` Format:**
    * Saves the model, including its architecture, weights, and optimizer state, into a single HDF5 file (`keras_model.h5`).
    * Loaded using `tf.keras.models.load_model()`.

* **TensorFlow `SavedModel` Format:**
    * Saves the model as a directory (`saved_model_tf/`) containing the graph definition (`.pb` file), variables, and assets.
    * This is the standard, universal format used for serving models in production.
    * Also loaded using `tf.keras.models.load_model()`.

### 2. TensorFlow Serving (Production Server)

This section demonstrates how to serve a model using **TensorFlow Serving**, a high-performance system designed for production environments.

* **Setup:** Shows the `bash` commands to pull the official `tensorflow/serving` Docker image.
* **Running the Server:** A Docker command is used to launch the serving container, pointing it at the `SavedModel` directory (which is versioned, e.g., `mnist_model/1/`).
* **Inference:**
    * An example script using the `requests` library formats a test image into a JSON payload.
    * It sends a **POST request** to the TF Serving REST API endpoint (`http://localhost:8501/v1/models/mnist_model:predict`) to get a live prediction from the served model.

### 3. TensorFlow Lite (On-Device)

This section covers converting a model for use on mobile and embedded devices using **TensorFlow Lite (TFLite)**, which creates smaller, faster, and more efficient models.

* **Conversion:**
    * A `tf.lite.TFLiteConverter` is initialized from a Keras model.
    * The `converter.convert()` method is called to produce the optimized model.
    * The resulting model is saved as a `.tflite` file (`model.tflite`).
* **Inference with Interpreter:**
    * The `.tflite` model is loaded into a `tf.lite.Interpreter`.
    * The interpreter is used to allocate tensors, set the input, invoke inference, and retrieve the final prediction.

### 4. TensorFlow.js (Web Browser)

This section demonstrates how to convert a model to run directly in a web browser (client-side) using **TensorFlow.js**.

* **Conversion:**
    * Shows the `bash` command to install the `tensorflowjs` converter.
    * The `tensorflowjs_converter` command-line tool is used to convert a Keras `.h5` file into a `tfjs_model/` directory, which contains a `model.json` file (for the architecture) and binary shard files (for the weights).
* **Inference in JavaScript:**
    * Provides an example **HTML/JavaScript snippet** that:
        1.  Includes the TensorFlow.js library.
        2.  Asynchronously loads the model using `tf.loadLayersModel('tfjs_model/model.json')`.
        3.  Runs a prediction using `model.predict()`.

---

## 🛠️ Requirements

To run this notebook and its examples, you will need the following:

* `tensorflow`
* `numpy`
* `requests`
* `json`
* `os`
* **Docker:** Required to run the TensorFlow Serving example.
* **`tensorflowjs`:** The Python command-line tool, installed via `pip install tensorflowjs`.

## 🚀 How to Use

You can run the `notebook_16.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
