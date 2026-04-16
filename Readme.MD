# Cat or Dog ML

## Overview

This project builds a complete machine learning pipeline to classify images of cats and dogs using classical machine learning techniques (no deep learning yet).

The project includes:

* Feature extraction using HOG (Histogram of Oriented Gradients) 
* Large-scale dataset processing using chunked pipelines
* Model benchmarking across multiple algorithms (work in progres)
* A GUI application for real-time predictions (work in progres)

---

## Goal of the Project

* Build a fully functioning application where a user can drag and drop an image of a cat or dog into a GUI, and the trained model predicts the result.
* Create a report with graphs and explanations covering the reasoning behind design decisions and model choices.

---

## What I Hope to Gain

* Learn more about the tuning process in machine learning
* Understand limitations and trade-offs between different models
* Explore performance vs efficiency in real-world scenarios
* Gain experience building a full ML pipeline from data processing to deployment

---

## Dataset

* Source: Kaggle PetImages dataset
* Size: ~24,000 images (cats and dogs)
* Preprocessing:

  * Images resized to 300x280
  * Converted to grayscale
  * HOG feature extraction (~44,000 features per image)
  * Invalid/corrupted images removed

---

## Pipeline

The project follows a structured pipeline:

1. Image validation and filtering
2. Feature extraction using HOG,LBP, HOG+LBP
3. Dataset creation (CSV format with chunked processing for memory efficiency)
4. Dataset shuffling and balancing
5. Train/test split with stratification
6. Model training using cross-validation
7. Model comparison and ranking
8. Model saving and deployment

---

## Model Evaluation

Models are evaluated using multiple metrics:

* **Score (F1 Score)** → model performance
* **Time** → training efficiency
* **Feature_type** → Feature extraction method
* **Warnings** → reliability / convergence issues

### Derived Metrics

* **Score/Time ratio** → performance efficiency
* **Robust Score** → penalizes unstable models
* **Baseline Gain** → improvement over baseline

---

## Baseline

A baseline model using a `DummyClassifier` is included.

This ensures that all trained models outperform trivial predictions and provides a meaningful reference point for evaluation.

---

## Model Selection Strategy

Three types of models are selected:

1. **Best Model (Accuracy)**

   * Highest F1 score

2. **Most Efficient Model (Speed)**

   * Best performance relative to training time

3. **Baseline Model**

   * Reference for comparison

This ensures a balance between performance, efficiency, and practicality.

---

## Application (GUI)

The final model is integrated into a graphical user interface where users can:

* Drag and drop an image
* Receive a prediction (cat or dog)

This demonstrates how the model can be used in a real-world scenario.

---

## Results

Example comparison:

| Model         | Score | Time | Std  |
| ------------- | ----- | ---- | ---- |
| SVC           | 0.85  | 12s  | 0.02 |
| Random Forest | 0.82  | 2s   | 0.01 |
| SGD           | 0.78  | 0.3s | 0.03 |

---

## Key Insights

* Higher accuracy models are often significantly slower
* Some models show instability across folds
* Simpler models can perform competitively when optimized
* High feature dimensionality (~44k features) impacts training speed
* Model reliability is as important as raw performance

---

## Future Work

* Hyperparameter tuning for top models
* Dimensionality reduction (e.g., PCA)
* Experiment with deep learning (CNNs)
* Improve GUI design and usability
* Optimize feature extraction pipeline

---

## Tech Stack

* Python
* scikit-learn
* OpenCV
* scikit-image
* pandas / numpy
* Tkinter (GUI)

---

## Summary

This project focuses on building a complete machine learning workflow while exploring the trade-offs between accuracy, speed, and reliability.

The goal is not just to achieve the best score, but to understand what makes a model practical in real-world applications.
