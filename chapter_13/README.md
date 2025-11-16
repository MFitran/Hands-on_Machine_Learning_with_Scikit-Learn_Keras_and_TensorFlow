# Chapter 13: Loading and Preprocessing Data with TensorFlow

This notebook serves as a summary and submission for **Chapter 13** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."*

This chapter moves beyond simple in-memory data loading and focuses on building efficient, scalable, and production-ready data pipelines using TensorFlow's powerful tools.

---

## 📚 Key Concepts Summarized

This notebook reproduces the code and theoretical explanations for the following key data pipeline components:

### 1. The `tf.data` API
* **Objective:** To build high-performance data pipelines by creating a `Dataset` object and chaining transformation methods.
* **Key Transformations:**
    * **Reading from files:** `tf.data.TextLineDataset` (for text) and `tf.data.FixedLengthRecordDataset` (for binary).
    * **Shuffling:** `dataset.shuffle()`.
    * **Batching:** `dataset.batch()`.
    * **Mapping (Preprocessing):** `dataset.map()` to apply custom preprocessing functions.
    * **Prefetching:** `dataset.prefetch()` to prepare the next batch of data while the model is training on the current one.

### 2. The TFRecord Format
* **Objective:** TensorFlow's preferred binary format for storing large datasets efficiently.
* **Process:** The notebook covers the full workflow:
    1.  Creating a `TFRecordDataset`.
    2.  Writing data to `.tfrecord` files using `tf.io.TFRecordWriter`.
    3.  Reading, parsing, and decoding the TFRecord files.
* **`tf.train.Example`:** Explains how to use this flexible protocol buffer to store complex features (like byte strings, float lists, or int lists) within a TFRecord file.

### 3. Keras Preprocessing Layers
* **Objective:** To embed preprocessing logic directly into the model. This prevents training-serving skew by ensuring the same preprocessing (e.g., normalization) is applied during both training and inference.
* **Layers Covered:**
    * `keras.layers.Normalization`: Computes and stores the mean and standard deviation of the training data.
    * `keras.layers.TextVectorization`: Handles text tokenization, vocabulary building, and encoding.

### 4. TensorFlow Datasets (TFDS)
* **Objective:** A high-level API to easily download and load over 100 common datasets (e.g., MNIST, CIFAR10).
* **Usage:** Demonstrates how to use `tfds.load()` to fetch a dataset, which can then be passed directly to a Keras model's `fit()` method.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `tensorflow` (version 2.0 or later)
* `keras` (bundled with TensorFlow)
* `tensorflow_datasets` (`tfds`)
* `scikit-learn`
* `numpy`
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install tensorflow tensorflow-datasets scikit-learn numpy jupyter
