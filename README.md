# Wine Quality Prediction

This project focuses on predicting the quality of wines based on their physicochemical properties. The goal is to classify wines into two categories: "good" (quality ≥ 6) and "bad" (quality < 6).

## Dataset Overview

- **Total Samples**: 6497 wine samples  
  - **Red wines**: 1599 samples  
  - **White wines**: 4898 samples  
- **Features**: 11 numeric physicochemical attributes (e.g., acidity, alcohol, sulfur dioxide)  
- **Target**: Quality score (integer between 3 and 9)  

The project uses this dataset to predict whether a wine is "good" or "bad" based on its features.

## Data Preparation

1. **Data Cleaning**: 
   - Removed duplicate entries (240 for red wine, 937 for white wine).
   - Ensured no missing values in the dataset.

2. **Class Imbalance Handling**: 
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the "good" and "bad" classes in the training set.

3. **Feature Scaling**: 
   - Used **StandardScaler** to standardize the features (mean = 0, std = 1).
   - Applied **Box-Cox PowerTransformer** to stabilize variance and reduce skewness.
   - **Winsorization** capped extreme values at the 1st and 99th percentiles.

## Model Implementation

1. **Logistic Regression (LR)**: 
   - Implemented both linear and polynomial feature versions of LR.
   - Polynomial feature mapping was applied to capture non-linear relationships.

2. **Support Vector Machine (SVM)**: 
   - Implemented both linear and polynomial kernel SVM models.
   - Polynomial kernel was used to handle non-linear decision boundaries.

3. **Cross-validation**: 
   - Used **5-fold cross-validation** to assess model performance and avoid overfitting.

## Model Evaluation

- **Logistic Regression (poly)**:  
  - **Accuracy**: 0.76  
  - **Precision**: 0.84  
  - **Recall**: 0.75  
  - **F1 Score**: 0.80  

- **SVM (poly)**:  
  - **Accuracy**: 0.74  
  - **Precision**: 0.86  
  - **Recall**: 0.70  
  - **F1 Score**: 0.78  

## Key Findings

- **Logistic Regression with polynomial features** was the best-performing model, with the highest accuracy and well-balanced evaluation metrics.
- Handling class imbalance and scaling features were crucial steps in improving the model's performance.
- Polynomial features were essential for capturing the non-linear relationships in the data, which linear models couldn't model effectively.

## Conclusion

This project demonstrated the importance of data preprocessing, handling class imbalance, and scaling features to improve model performance. Using polynomial features allowed linear models like Logistic Regression to handle more complex relationships, resulting in better predictions. This project provided practical insights into machine learning techniques and their application to real-world problems.

---

### Requirements

- Python 3.x
- Libraries:  
  - pandas  
  - numpy  
  - scikit-learn  
  - imbalanced-learn  
  - matplotlib (optional for visualizations)

### Usage

1. Clone the repository.
2. Install the required dependencies:
