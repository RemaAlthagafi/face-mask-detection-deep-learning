# Face Mask Detection Using Deep Learning

A deep learning project that aims to classify facial images into two categories: **with mask** and **without mask**.

## Status
**Work in Progress**  
This project is currently under development. The repository will be updated with implementation, experiments, results, and final documentation.

## Project Overview
This project focuses on building a face mask classification system using deep learning techniques.  
The proposed approach uses transfer learning to improve classification performance on facial images under different real-world conditions such as lighting variation, face orientation, and mask appearance.

## Objectives
- Build a deep learning model to classify masked and unmasked faces
- Train and improve the model using labeled image data
- Evaluate model performance using standard metrics
- Analyze the effectiveness of the model in real-world image conditions

## Planned Methodology
The project is planned to use:
- **CNN-based image classification**
- **Transfer Learning**
- **MobileNetV2** as a pre-trained base model

### Pipeline
1. Image preprocessing
2. Data augmentation
3. Feature extraction using MobileNetV2
4. Classification using fully connected layers
5. Training and evaluation

## Dataset
The project is planned to use the **Face Mask 12K Images Dataset** from Kaggle, which contains labeled images for two classes:
- With Mask
- Without Mask

## Evaluation Metrics
The model performance will be evaluated using:
- Accuracy
- F1-score
- Confusion Matrix
- Training and Validation Accuracy/Loss Curves


## Future Work
The next steps include:
- Implementing the training pipeline
- Fine-tuning the model
- Comparing optimization techniques
- Evaluating results and documenting findings