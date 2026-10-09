# Heart Disease Classification

Machine learning-based heart disease classification using the UCI Heart Disease dataset. This project includes data preprocessing, model training, performance evaluation, and comparative analysis of multiple supervised learning algorithms.

## Overview

Heart disease is one of the leading causes of death worldwide, making early diagnosis an important healthcare challenge. In this project, supervised machine learning techniques are applied to classify patients based on clinical and demographic attributes from the UCI Heart Disease dataset.

The project focuses on:

- Data preprocessing and preparation
- Training multiple classification models
- Performance evaluation using standard classification metrics
- Comparative analysis of model behavior and limitations

## Dataset

**Dataset:** UCI Heart Disease Dataset

The dataset contains medical attributes commonly used in heart disease diagnosis, including demographic information, clinical measurements, and diagnostic indicators.

## Methods

### Data Preprocessing

- Handling missing values
- Feature selection
- Data cleaning
- Data normalization/scaling
- Preparation of training and testing datasets

### Classification Models

The following supervised learning algorithms were implemented and evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)

## Evaluation Metrics

Model performance was evaluated using the following classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics provide a comprehensive assessment of each model's ability to correctly classify heart disease cases.

## Results

### Logistic Regression

- Highest overall performance among the evaluated models
- Better identification of healthy patients (Class 0)
- Limited ability to distinguish advanced disease classes

### Support Vector Machine (SVM)

- Comparable performance to Logistic Regression
- Strong performance on the majority class
- Difficulty recognizing minority disease classes

### K-Nearest Neighbors (KNN)

- Similar overall accuracy to SVM
- Slightly improved performance on some intermediate disease classes
- Reduced effectiveness for severe disease categories

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 55%      | 48%       | 55%    | 50%      |
| SVM                 | 52%      | 42%       | 52%    | 46%      |
| KNN                 | 52%      | 45%       | 52%    | 47%      |

## Discussion

The experimental results indicate that all models perform reasonably well in identifying healthy patients but struggle to classify less frequent disease categories. This behavior suggests the presence of class imbalance within the dataset.

Among the evaluated algorithms, Logistic Regression achieved the best overall performance. However, none of the models provided strong classification results for advanced heart disease classes, indicating potential opportunities for improvement through data balancing techniques, feature engineering, or more advanced machine learning models.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Key Learning Outcomes

- Data preprocessing for healthcare datasets
- Supervised machine learning classification
- Model evaluation and comparison
- Interpretation of classification metrics
- Analysis of class imbalance issues in medical data

## Documentation

A brief project report describing the implementation process, model training, and inference workflow is available in:

```text
Report.pdf
```
