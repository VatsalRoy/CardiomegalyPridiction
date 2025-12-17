# Cardiomegaly Detection in Chest X-rays using a Convolutional Neural Network

> **GitHub Repositor**y: [CardiomegalyPridiction](https://github.com/VatsalRoy/CardiomegalyPridiction)

> **Data Source:** [NIH CheXpert chest X-ray collection](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)

<p align="center"> <img src="report/projectPoster.png" alt="Cardiomegaly Detection Project Poster" width="1080"> </p>

***

## Project Overview and Motivation

Cardiomegaly, or abnormal enlargement of the heart, is a key radiological marker for cardiovascular diseases (CVDs), which account for roughly one-third of global deaths annually.  

Traditionally, cardiomegaly diagnosis relies on clinicians manually assessing CXRs and computing the Cardiothoracic Ratio (CTR), a process that is time-consuming, prone to diagnostic delays in high-volume settings, and subject to inter-observer variability based on clinical experience.  

This project leverages deep learning and Convolutional Neural Networks (CNNs) to build an automated, precise, and scalable screening tool, with a primary objective of designing, training, and rigorously evaluating a custom CNN for binary classification of cardiomegaly vs. normal CXRs.  

***

## Methodology

A four-step machine learning pipeline was implemented, covering data curation, preprocessing, model design, and evaluation.

### -> Data Curation and Preparation

- **Dataset source:** A specialized subset was curated from the NIH CheXpert collection.  
- **Final dataset size:** 5,447 chest X-ray images after quality assurance filtering.  
- **Preprocessing:**  
  - Images resized to \(224 \times 224\) pixels.  
  - Pixel intensities normalized to the \([0, 1]\) floating-point range.  
- **Data split (stratified):**  
  - 70% training  
  - 20% validation  
  - 10% testing  

### -> Model Architecture and Training

- **Feature extractor:**  
  - 4 sequential convolutional blocks with filters increasing from 32 to 256.  
  - Each block: 3×3 convolution, ReLU, batch normalization, 2×2 max-pooling, and 25% dropout.  
- **Classifier head:**  
  - Two fully connected layers with 512 and 256 neurons, each using 50% dropout for regularization.  
- **Model size:**  
  - Total of 26,215,041 trainable parameters.  
- **Training setup:**  
  - Optimizer: Adam with learning rate 0.0001.  
  - Loss: Binary cross-entropy.  
  - Early stopping applied over 56 epochs to mitigate overfitting.  

***

## Performance and Results

The model was evaluated on the held-out test set with the following metrics:

| Metric                               | Value  | Notes |
|--------------------------------------|--------|-------|
| Overall accuracy                     | 79.8%  | Solid baseline performance but below SOTA. |
| Area Under ROC Curve (AUC)          | 0.906  | Indicates strong capability to distinguish positive vs. negative cases. |
| Cardiomegaly precision               | 0.86   | 86% of samples predicted positive truly have cardiomegaly. |
| Cardiomegaly recall (sensitivity)    | 0.71   | Only 71% of actual cardiomegaly cases detected. |

**Clinical limitation:**  
A recall of 0.71 implies roughly 29% of actual cardiomegaly cases are missed (31 false negatives out of 269 true positives in the test set). For clinical decision support, sensitivity typically needs to exceed 90%, making this model unsuitable for deployment but useful as a research baseline.

***

## Comparison

The custom CNN forms a strong baseline but underperforms compared to leading models in the literature.

- Transfer-learning-based models (e.g., ResNet-50) report accuracies up to **99.8%** on similar cardiomegaly detection tasks.  
- Segmentation-based approaches (e.g., U-Net) achieve around **94%** accuracy.  
- The performance gap of approximately **15–20 percentage points** is largely due to:
  - The relatively shallow, from-scratch architecture (no transfer learning).  
  - Limited architectural complexity compared to deep residual or dense networks.  

***

## 5. Future Work

To close the gap with state-of-the-art performance and reach clinically viable sensitivity:

- **Transfer learning:**  
  - Fine-tune pre-trained architectures such as ResNet-50, DenseNet, or EfficientNet on the curated cardiomegaly dataset.  
- **Data augmentation:**  
  - Apply advanced augmentations (e.g., elastic transforms, contrast-limited adaptive histogram equalization) to improve generalization.  
- **Ensemble methods:**  
  - Combine predictions from multiple architectures to enhance robustness and stability.  
- **Threshold optimization:**  
  - Adjust the classification threshold to prioritize higher recall, tuned to the requirements of specific clinical workflows.  

***

## Authors

Developed by **Vatsal Roy**, **Kunj Patel**, **Riya Gardharia**, and **Shreya Vaghela**.
