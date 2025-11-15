# Chapter 11: Training Deep Neural Networks

This notebook serves as a summary and submission for **Chapter 11** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

The chapter tackles the primary challenges encountered when training very deep neural networks (e.g., 10+ layers) and provides the modern solutions that make Deep Learning practical.

---

## 📚 Chapter Summary: Key Problems and Solutions

This notebook explores the main problems in training DNNs and reproduces the code for their solutions.

### 1. The Vanishing/Exploding Gradients Problem
Gradients can become extremely small (vanish) or large (explode) as they are backpropagated through many layers, making it impossible for lower layers to train.

**Solutions Covered:**
* **Initialization:** Using smarter weight initialization, like **Glorot/Xavier** and **He initialization**.
* **Activation Functions:** Using non-saturating functions like **ReLU** and its variants (Leaky ReLU, ELU, PReLU).
* **Batch Normalization:** Adding `BatchNormalization` layers to normalize the inputs to each layer, which stabilizes training and acts as a regularizer.
* **Gradient Clipping:** Clipping the gradients during optimization to prevent them from exploding.

### 2. Slow Training
Training a very large network can be extremely slow.

**Solutions Covered:**
* **Transfer Learning:** Reusing the lower layers of a pretrained network (e.g., one trained on ImageNet) to avoid training a large network from scratch.
* **Faster Optimizers:** Replacing standard Stochastic Gradient Descent with more advanced optimizers like **Momentum**, **Nesterov Accelerated Gradient (NAG)**, **AdaGrad**, **RMSProp**, **Adam**, and **Nadam**.
* **Learning Rate Scheduling:** Using techniques to automatically adjust the learning rate during training, such as power, exponential, piecewise, and performance scheduling.

### 3. Overfitting
With millions of parameters, deep networks are highly prone to overfitting the training data.

**Solutions Covered:**
* **Regularization:** Using $\ell_1$ and $\ell_2$ regularization.
* **Dropout:** Adding `Dropout` layers, which randomly "turn off" a percentage of neurons during each training step, forcing the network to learn more robust features.
* **Monte Carlo (MC) Dropout:** Using dropout during inference to get multiple predictions and a measure of uncertainty.
* **Max-Norm Regularization:** Constraining the maximum weight value for each neuron.

---

## 💻 Code & Concepts Reproduced

This notebook provides code examples for the following techniques:

* **Initialization:** `keras.layers.Dense(..., kernel_initializer="he_normal")`.
* **Activation Functions:** `keras.layers.LeakyReLU(alpha=0.2)`.
* **Batch Normalization:** `keras.layers.BatchNormalization()`.
* **Gradient Clipping:** `optimizer=keras.optimizers.SGD(clipvalue=1.0)`.
* **Transfer Learning:** Loading weights (`load_weights`) and freezing layers (`layer.trainable = False`).
* **Optimizers:**
    * `keras.optimizers.SGD(lr=0.001, momentum=0.9)`
    * `keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)`
    * `keras.optimizers.Adagrad(lr=0.001)`
    * `keras.optimizers.RMSprop(lr=0.001, rho=0.9)`
    * `keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)`
* **Learning Rate Scheduling:**
    * Power: `optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-4)`
    * Exponential and Piecewise: Using a custom callback or function.
    * Performance: `keras.callbacks.ReduceLROnPlateau`.
* **Regularization:**
    * $\ell_2$: `keras.layers.Dense(..., kernel_regularizer=keras.regularizers.l2(0.01))`
    * Dropout: `keras.layers.Dropout(rate=0.2)`
    * Max-Norm: `kernel_constraint=keras.constraints.max_norm(1.)`

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `tensorflow` (version 2.0 or later)
* `keras` (bundled with TensorFlow)
* `scikit-learn`
* `numpy`
* `matplotlib`
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install tensorflow scikit-learn numpy matplotlib jupyter
