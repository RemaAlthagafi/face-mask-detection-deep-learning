# Face Mask Detection Using Deep Learning (Transfer Learning)

An advanced deep learning repository dedicated to classifying facial images into two primary categories: **With Mask** and **Without Mask**. This project leverages Transfer Learning with a customized architecture to achieve high stability, fast convergence, and robust generalization in human-centric safety monitoring.

---

## 📌 Project Overview & Objectives
The core goal of this project is to develop a reliable, highly accurate computer vision system capable of identifying whether an individual is properly wearing a face mask in real-world image conditions.
- **Core Objective:** Build a solid, production-ready Baseline Model using Transfer Learning.
- **Data Optimization:** Implement a controlled data sampling pipeline to balance computational efficiency and prevent data leakage.
- **Hyperparameter Analysis:** Evaluate multiple independent optimization techniques (Dropout, Label Smoothing, Batch Normalization) exclusively across the validation and testing phases.

---

## 🛠️ System Architecture & Methodology

Instead of training a deep convolutional neural network (CNN) from scratch, this project utilizes **Transfer Learning** to exploit features already learned by a robust, deep architecture.

### 1. The Pre-trained Base (Feature Extractor)
We utilized a **ResNet50** backbone pre-trained on the comprehensive **DeepWeeds** image repository (`resnet.hdf5`). 
- The top classification layers of the ResNet50 model were completely removed (`include_top=False`), leaving the frozen 50-layer deep convolutional structure as a highly efficient feature extractor. It is capable of recognizing complex edge distributions, facial boundaries, and structural variations.
- All base layers were explicitly frozen (`layer.trainable = False`) to preserve the pre-trained weights during the training process.

### 2. Custom Classifier Head
A newly designed, lightweight classification head was appended directly to the frozen ResNet50 base:
- **Global Average Pooling 2D:** Used to reduce spatial dimensions and minimize overall parameter density.
- **Dense Layer (Softmax):** A final fully connected layer with 2 output units corresponding to the target classes (`With Mask` and `Without Mask`).

---

## 📊 Pipeline & Implementation Steps

1. **Environment Setup:** Integrated with `tf_keras` execution environments to handle backward-compatibility and ensure consistent performance.
2. **Data Sampling & Balancing:** Out of the complete dataset, an automated pipeline cleans, resets, and extracts a controlled subset of **2000 balanced images per class** for training, alongside dedicated isolated partitions for validation and testing.
3. **Data Pipelines:** Utilizing `ImageDataGenerator` with pixel rescaling (`1./255`). Shuffling is strictly disabled (`shuffle=False`) during evaluation to guarantee correct Confusion Matrix alignment.
4. **Controlled Training & Callbacks:** - Training is monitored via matching step allocations ($63$ steps for training, $25$ steps for validation).
   - Integrated `EarlyStopping` to halt training when validation loss stabilizes, and `ModelCheckpoint` to save the optimal model weight configuration state (`best_model.keras`).

---

## 📈 Baseline Experimental Results

The Baseline model converged rapidly due to the robust initialization of the pre-trained features:
- **Training Accuracy:** Reached `100%` by the final epoch.
- **Validation Accuracy:** Stabilized at an exceptional **`98.25%`**.
- **Test Accuracy:** Final unbiased evaluation on the independent Test partition yielded **`98.89%`**.

### Performance Analysis
The loss and accuracy curves demonstrate clean convergence with a minimal, healthy gap between training and validation trends, showing absolute stability against overfitting.

---

## 🚀 Hyperparameter Tuning & Improvements
To further analyze the system, multiple hyperparameter optimization strategies can be evaluated on top of the established Baseline framework, validating results against the validation and testing partitions:

* **Improvement A (Dropout Integration):** Introducing a `layers.Dropout(0.5)` layer prior to classification to introduce regularization by randomly deactivating nodes during weight updates.
* **Improvement B (Label Smoothing Regularization):** Modifying the cross-entropy objective function by applying `label_smoothing=0.1` to prevent over-confidence in predictions and increase resilience to noisy visual bounds.
* **Improvement C (Batch Normalization):** Injecting explicit feature normalization layers after spatial pooling to stabilize internal covariate shifts within the customized classification head.

---

## 📂 Repository Structure
```text
├── mask_project.ipynb        # Comprehensive Jupyter Notebook with all execution cells
├── best_model.keras          # The finalized saved baseline model weights
└── README.md                 # Project documentation report
