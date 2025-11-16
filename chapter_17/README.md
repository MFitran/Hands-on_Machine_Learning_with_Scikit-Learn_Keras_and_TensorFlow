# Chapter 17: Representation Learning and Generative Learning Using Autoencoders and GANs

This notebook contains the code reproductions and theoretical explanations for Chapter 17 of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

---

## 📚 Chapter Summary

This chapter explores two powerful unsupervised learning techniques: **Autoencoders** and **Generative Adversarial Networks (GANs)**. Both are capable of learning dense, compact representations (codings) of data, but they work in very different ways.

### 1. Autoencoders
Autoencoders are neural networks that learn to reconstruct their own inputs. They consist of two parts:
* An **Encoder** that compresses the input data into a low-dimensional latent representation (the "coding").
* A **Decoder** that reconstructs the original data from this coding.

By placing constraints on the network (like a small coding layer or noisy inputs), the autoencoder is forced to learn the most important features in the data.

**Autoencoder architectures covered in this notebook:**
* **Linear Autoencoder:** Demonstrates how a simple linear autoencoder can perform Principal Component Analysis (PCA).
* **Stacked Autoencoders:** Deep autoencoders with multiple hidden layers, used for dimensionality reduction and visualization (e.g., on Fashion MNIST).
* **Tying Weights:** A technique to halve the model's parameters by making the decoder's weights the transpose of the encoder's weights, implemented with a custom `DenseTranspose` layer.
* **Convolutional Autoencoders:** Using `Conv2D` layers for the encoder and `Conv2DTranspose` layers for the decoder to efficiently handle image data.
* **Denoising Autoencoders:** Training an autoencoder to recover a clean input from a corrupted one (e.g., by adding a `Dropout` layer to the input).
* **Sparse Autoencoders:** Forcing the coding layer to be sparse (mostly zeros) to learn more specific features, using either `ActivityRegularization` ($\ell_1$) or a custom `KLDivergenceRegularizer`.
* **Variational Autoencoders (VAEs):** A **generative** autoencoder that learns a probability distribution of the data. The encoder outputs a mean (μ) and log variance (γ), from which a coding is sampled. This allows the model to generate new instances that look similar to the training data.

### 2. Generative Adversarial Networks (GANs)
GANs are a different type of generative model composed of two competing networks:
* A **Generator** that creates fake data (e.g., images) from random noise.
* A **Discriminator** that acts as a classifier, trying to distinguish the fake data from real data.

The two networks are trained in a zero-sum game, where the generator gets better at fooling the discriminator, and the discriminator gets better at spotting fakes, pushing the generator to create increasingly realistic data.

**GAN concepts covered in this notebook:**
* **GAN Training:** Implementing the two-phase custom training loop where the discriminator and generator are trained alternately.
* **Training Difficulties:** Discussing the main challenges of GAN training, such as **mode collapse** and **training instability**.
* **Deep Convolutional GANs (DCGANs):** Reproducing the DCGAN architecture, which provides a stable set of guidelines for building GANs using strided convolutions (`Conv2D`) in the discriminator and transposed convolutions (`Conv2DTranspose`) in the generator.
* **Advanced GANs:** Briefly explaining the concepts behind **Progressive Growing of GANs (ProGAN)** and **StyleGAN**.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `tensorflow` (version 2.0 or later)
* `keras` (bundled with TensorFlow)
* `scikit-learn` (for `StandardScaler`)
* `numpy`
* `matplotlib`
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install tensorflow scikit-learn numpy matplotlib jupyter
