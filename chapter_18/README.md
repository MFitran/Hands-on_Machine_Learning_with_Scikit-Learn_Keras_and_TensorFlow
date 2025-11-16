# Chapter 14: Deep Computer Vision Using Convolutional Neural Networks

This notebook contains the code reproductions and theoretical explanations for Chapter 14 of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

This chapter introduces **Convolutional Neural Networks (CNNs)**, the architecture that powers modern computer vision.

---

## 📚 Chapter Summary: Key Concepts

This notebook explores the fundamental building blocks of CNNs, from their biological inspiration to advanced applications like object detection and segmentation.

Key concepts covered include:

* **The Visual Cortex:** Understanding the biological inspiration for CNNs, particularly the concept of local receptive fields.
* **Convolutional Layers (`Conv2D`):** The core building block of CNNs. This section explains how they work using **filters** (or kernels) to detect local patterns. It also covers key hyperparameters like **padding** (`"valid"` vs. `"same"`) and **stride**.
* **Pooling Layers (`MaxPooling2D`):** Layers used to subsample (shrink) the feature maps, which reduces computational load, memory usage, and helps prevent overfitting. The notebook covers **max pooling** and **global average pooling**.
* **CNN Architectures:** Explanations of how convolutional and pooling layers are assembled to build deep networks. This includes discussions of modern architectures like **ResNet** and the concept of residual (skip) connections.
* **Transfer Learning:** A practical technique that involves reusing the lower layers of a **pretrained model** (like one trained on ImageNet) to build a powerful classifier without needing a massive dataset.

---

## 💻 Code & Concepts Reproduced

This notebook provides code examples and theoretical explanations for the following topics:

### 1. Building a Simple CNN
* **Dataset:** Fashion MNIST
* **Model:** A `Sequential` Keras model is built from scratch to demonstrate the standard CNN architecture:
    * `Conv2D`
    * `MaxPooling2D`
    * `Flatten`
    * `Dense` (for the final classification head)

### 2. Using Pretrained Models (Transfer Learning)
* **Model:** Demonstrates how to load a pretrained **ResNet50** model from `keras.applications`.
* **Technique:** The base layers of ResNet50 are frozen (`base_model.trainable = False`), and a new classification head (consisting of `GlobalAveragePooling2D` and `Dense` layers) is added on top. This new model is then trained on the target dataset (e.g., Fashion MNIST).

### 3. Advanced Computer Vision Tasks
The notebook also provides theoretical explanations for more advanced tasks:

* **Object Detection:** Moving beyond simple classification to identify *where* an object is in an image using **bounding boxes**. This section explains the concept of **Non-Max Suppression** to clean up multiple redundant predictions.
* **Semantic Segmentation:** The task of classifying *every pixel* in an image (e.g., all pixels belonging to a car are one color, all pixels for a road are another). This introduces the concept of **upsampling** using **transposed convolutional layers** (`Conv2DTranspose`) to build an encoder-decoder architecture.

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
