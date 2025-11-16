# Chapter 15: Processing Sequences Using RNNs and CNNs

This notebook contains the code reproductions and theoretical explanations for Chapter 15 of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

---

## 📚 Chapter Summary: Key Concepts

This chapter introduces **Recurrent Neural Networks (RNNs)**, a class of neural networks designed to handle sequential data like time series or text. Unlike feedforward networks, RNNs have connections that point backward, allowing them to maintain a form of memory.

Key concepts covered include:

* **Recurrent Neurons and Layers:** The fundamental concepts of how an RNN cell works by maintaining a hidden state and how layers are "unrolled through time" to be trained.
* **Backpropagation Through Time (BPTT):** The algorithm used to train RNNs, which is essentially backpropagation applied to the unrolled network.
* **Time Series Forecasting:** Building RNNs to predict future values in a sequence.
* **Handling Long Sequences:** The notebook explores two main challenges of training RNNs on long sequences:
    1.  **Unstable Gradients (Vanishing/Exploding):** Solved with techniques like Layer Normalization and recurrent dropout.
    2.  **Limited Short-Term Memory:** The inability of simple RNNs to capture long-term dependencies.
* **Advanced RNN Cells:** To solve the memory problem, two powerful cell architectures are introduced:
    * **Long Short-Term Memory (LSTM):** A complex cell that uses gates (forget, input, output) to manage both a short-term state (**h**) and a long-term state (**c**).
    * **Gated Recurrent Unit (GRU):** A simplified and more efficient version of the LSTM that often performs just as well.
* **CNNs for Sequences:** The chapter also explores using **1D convolutional layers** to process sequences, which can be much faster than RNNs. This culminates in the **WaveNet** architecture, which uses dilated 1D convolutions to efficiently learn very long-term patterns.

---

## 💻 Code & Concepts Reproduced

This notebook provides code examples and theoretical explanations for the following topics:

* **Generating Time Series Data:** Creating a synthetic time series dataset to use for forecasting.
* **Establishing Baselines:** Creating a naive forecasting (last value) model and a simple linear (Dense) model to measure baseline performance.
* **Simple RNN:** Building and training a `keras.layers.SimpleRNN` model.
* **Deep RNNs:** Stacking multiple recurrent layers, making sure to use `return_sequences=True` on all but the last layer.
* **Multi-Step Forecasting:** Demonstrating three different ways to predict multiple time steps ahead:
    1.  **Iterative Forecasting:** Using a single-step model and feeding its own predictions back as input.
    2.  **Sequence-to-Vector:** Training a model with a `Dense(10)` output layer to predict all 10 steps at once.
    3.  **Sequence-to-Sequence:** Using `TimeDistributed(keras.layers.Dense(10))` to make the RNN predict a 10-step forecast at every time step, which stabilizes training.
* **Handling Long Sequences (Advanced Cells):**
    * `keras.layers.LSTM`: Implementing a deep RNN using LSTM cells.
    * `keras.layers.GRU`: Implementing a deep RNN using GRU cells.
* **CNNs for Sequences:**
    * **CNN+RNN:** Using a `keras.layers.Conv1D` layer to downsample the sequence before feeding it to a GRU network, which is much faster.
    * **WaveNet:** Building a simple WaveNet-style model by stacking `Conv1D` layers with increasing `dilation_rate` and `padding="causal"` to handle long-term dependencies.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `tensorflow` (version 2.0 or later)
* `keras` (bundled with TensorFlow)
* `numpy`
* `matplotlib`
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install tensorflow numpy matplotlib jupyter
