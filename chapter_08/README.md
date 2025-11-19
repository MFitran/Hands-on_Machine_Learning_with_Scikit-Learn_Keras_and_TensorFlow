# Chapter 8: Neural Network Layers and Advanced Features

This Jupyter Notebook (`notebook_08.ipynb`) contains the code and theoretical summaries from **Chapter 8 of "TensorFlow in Action" by Thushan Ganegedara**.

This notebook serves as a deeper dive into the tools Keras offers for building production-ready models, focusing on specialized layers, proper weight initialization, and powerful regularization techniques necessary to train deep, complex networks effectively.

The primary examples use the **Fashion MNIST** and **IMDB** datasets.

---

##  summary-of-contents

### 1. Specialized Keras Layers

This section demonstrates the use of various advanced and specialized layers integrated into Keras:

* **Convolutional Layers:** Introduces the appropriate use cases for `Conv1D` (e.g., text/time series), `Conv2D` (images), and `Conv3D` (video/volumetric data).
* **Normalization Layers:**
    * **`BatchNormalization`**: Improves the stability and speed of training by normalizing the inputs of a layer across the batch dimension.
    * **`LayerNormalization`**: Normalizes inputs across the feature dimension rather than the batch, making it generally more suitable for Recurrent Neural Networks (RNNs) and **Transformers**.
* **Custom Activations:** Shows how to use any callable Python function (like `tf.nn.swish`) as a layer's activation function.

### 2. Weight Initializers, Regularizers, and Constraints

These tools are crucial for controlling the behavior of model weights during and after training:

* **Initializers:** Demonstrates how to prevent vanishing or exploding gradients by specifying initial weights using standard techniques like **`VarianceScaling`** (a generalized method including Glorot/Xavier and He initializations).
* **Regularizers (L1 & L2):** Shows how to penalize large weights by adding the L1/L2 norm of the weight vector to the loss function, encouraging the model to use simpler representations and preventing overfitting. This is done using **`tf.keras.regularizers.L1L2`**.
* **Constraints:** Introduces constraints that force weights to stay within certain bounds after each gradient update. The example uses **`tf.keras.constraints.MaxNorm`** to restrict the L2 norm of the weight vectors.

### 3. Advanced Model Techniques

This section brings multiple concepts together, culminating in the creation of a custom-gated neural network layer.

* **Model A (CNN with Advanced Components):** Builds a simple Convolutional Neural Network (CNN) for Fashion MNIST, incorporating many of the elements discussed (e.g., `BatchNormalization`, `Dropout`, and a custom activation) to show their integration in a standard architecture.
* **Model B (Custom Gated Layer):** Demonstrates the power of the **Sub-classing API** by creating a **`CustomGatedLayer`**. This layer manually implements a computation with an internal gating mechanism (similar to an LSTM cell), where one branch acts as a **gate** (an activation function applied to weights) that controls the flow of information from a second **transform** branch.

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:

* `tensorflow`
* `numpy`

## 🚀 How to Use

You can run the `notebook_08.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
