# Chapter 19: Training and Deploying TensorFlow Models at Scale

This notebook contains the code reproductions and theoretical explanations for Chapter 19 of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

---

## 📚 Chapter Summary

This chapter covers the crucial final steps in a machine learning project: deploying a trained model into a production environment and scaling up training for large, complex models.

Key topics covered include:

* **Serving a TensorFlow Model:** The notebook explains how to export a model to TensorFlow's **SavedModel** format, which is a universal, language-neutral format that includes the computation graph and weights.
* **Deploying with TensorFlow Serving (TF Serving):**
    * Demonstrates how to use **Docker** to install and run TF Serving, a high-performance, production-ready serving system.
    * Shows how to query a served model using both the **REST API** (simple, JSON-based) and the **gRPC API** (high-performance, binary-based).
* **Deploying to the Cloud:** Provides an overview of deploying models to cloud platforms like **Google Kubernetes Engine (GKE)** and **Google AI Platform**.
* **Distributed Training:** Discusses strategies for scaling up the training of large models, including using **TensorFlow's Distribution Strategies API** (like `MirroredStrategy`) to train on multiple GPUs.
* **MLOps (ML Pipelines):**
    * Introduces **TensorFlow Extended (TFX)** as a framework for building end-to-end ML production pipelines.
    * Briefly covers **MLflow** as an open-source alternative for experiment tracking, packaging code, and deploying models.

---

## 🔧 Key Technologies & Tools

This notebook serves as a guide to a wide array of production-level tools and concepts:

* TensorFlow Extended (TFX)
* TensorFlow Serving (TF Serving)
* TensorFlow Hub
* Docker
* gRPC (Google Remote Procedure Call)
* Google Kubernetes Engine (GKE)
* Google AI Platform (Vertex AI)
* MLflow
* TensorFlow's Distribution Strategies

---

## 🚀 How to Run

This notebook is primarily a conceptual guide and demonstration of deployment workflows. Fully running the code requires external tools and services.

**Prerequisites:**
* `tensorflow` (version 2.0 or later)
* `keras`
* `numpy`
* `grpcio`
* `tensorflow-serving-api`
* `docker` (must be installed and running on your system to use TF Serving locally)

You can install the Python dependencies using `pip`:
```bash
pip install tensorflow grpcio tensorflow-serving-api numpy
