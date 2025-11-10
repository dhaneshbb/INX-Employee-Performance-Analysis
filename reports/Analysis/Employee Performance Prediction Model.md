# Employee Performance Rating Prediction Model

**Report Date:** April 8, 2025
**Last Revised:** November 10, 2025

**Project Code:** 10281
**Document Code:** CDS_Project_2_INX_Future_Emp_Data_V1.6
**Prepared for:** INX Future Inc.

---

## Executive Summary

This report evaluates multiple machine learning models to predict employee performance ratings (Rating 2: Good, Rating 3: Excellent, Rating 4: Outstanding) using a dataset of 1,200 records with 11 features. The top-performing model, **XGBoost**, achieved **92.5% accuracy** with minimal overfitting (0.83% train-test gap) and effective handling of class imbalances. Key findings include:

- XGBoost achieved highest accuracy, F1-score (92.4%), and ROC AUC (0.976)
- EmpEnvironmentSatisfaction and EmpLastSalaryHikePercent were the most influential features across all models
- Ensemble methods (Voting/Stacking) achieved 92.08% and 91.25% accuracy respectively

---

## Table of Contents

- [Employee Performance Rating Prediction Model](#employee-performance-rating-prediction-model)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [1. Data Overview](#1-data-overview)
    - [1.1 Dataset Description](#11-dataset-description)
    - [1.2 Preprocessing](#12-preprocessing)
  - [2. Methodology](#2-methodology)
    - [2.1 Data Preparation](#21-data-preparation)
    - [2.2 Models Evaluated](#22-models-evaluated)
  - [3. Model Performance Comparison](#3-model-performance-comparison)
  - [4. Final Model: XGBoost](#4-final-model-xgboost)
    - [4.1 Hyperparameters](#41-hyperparameters)
    - [4.2 Performance Evaluation](#42-performance-evaluation)
  - [5. Feature Importance Analysis](#5-feature-importance-analysis)
  - [6. Discussion](#6-discussion)
  - [7. Recommendations](#7-recommendations)
  - [8. Conclusion](#8-conclusion)
  - [Appendix](#appendix)

---

## 1. Data Overview

### 1.1 Dataset Description

**Dataset Specifications:**
- **Size:** 1,200 employees, 12 features
- **Target Variable:** PerformanceRating (3 classes: 0=Good, 1=Excellent, 2=Outstanding)

**Key Features:**

**Job Satisfaction Metrics:**
- EmpEnvironmentSatisfaction
- EmpJobSatisfaction
- EmpWorkLifeBalance

**Experience and Tenure:**
- ExperienceYearsAtThisCompany
- TotalWorkExperienceInYears
- YearsSinceLastPromotion

**Compensation and Role:**
- EmpLastSalaryHikePercent
- EmpJobRole_freq

**Work Conditions:**
- OverTime
- DistanceFromHome

### 1.2 Preprocessing

**Class Mapping:**
- Target labels remapped to [0, 1, 2] for model compatibility
- Original: Rating 2 (Good) → 0, Rating 3 (Excellent) → 1, Rating 4 (Outstanding) → 2

**Stratified Split:**
- Training set: 960 samples (80%)
- Test set: 240 samples (20%)

**Class Balancing:**
- Weighted classes to address imbalance
- Class weights: Class 0 = 2.06, Class 1 = 0.46, Class 2 = 3.02

---

## 2. Methodology

### 2.1 Data Preparation

**Target Variable:**
- PerformanceRating (3 classes) mapped to [0, 1, 2] for compatibility

**Class Weights:**
- Applied to address imbalance
- Weights: 2.06 for Class 0, 0.46 for Class 1, 3.02 for Class 2

**Train-Test Split:**
- Stratified split (80-20) to preserve class distribution

### 2.2 Models Evaluated

**Base Models:**
- Logistic Regression
- Random Forest
- LightGBM
- CatBoost
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- Naive Bayes

**Ensemble Models:**
- Voting Classifier (soft voting)
- Stacking Classifier (Logistic Regression meta-model)

**Hyperparameter Tuning:**
- RandomizedSearchCV with 50 iterations
- 5-fold stratified cross-validation
- Scoring metric: f1_weighted

---

## 3. Model Performance Comparison

**Key Metrics on Test Set:**

| Model | Accuracy | F1-Score | ROC AUC | Training Time (s) | Overfit (Train - Test) |
|-------|----------|----------|---------|-------------------|-------------------------|
| **XGBoost (Tuned)** | 92.50% | 92.42% | 0.976 | 0.82 | 0.0083 |
| Random Forest (Tuned) | 92.50% | 92.34% | 0.973 | 0.83 | 0.0531 |
| Voting Ensemble | 92.08% | 91.99% | 0.974 | 3.84 | 0.0677 |
| Stacking Ensemble | 91.25% | 91.19% | 0.967 | 21.49 | 0.0552 |
| CatBoost (Tuned) | 90.83% | 90.80% | 0.970 | 0.66 | 0.0865 |
| LightGBM (Tuned) | 89.17% | 89.19% | 0.960 | 0.37 | 0.1083 |
| MLP Classifier (Tuned) | 82.92% | 82.11% | 0.899 | 25.75 | 0.1104 |

**Analysis:**
- XGBoost and Random Forest achieve highest test accuracy (92.50%)
- XGBoost shows minimal overfitting (0.83% gap) compared to Random Forest (5.31%)
- Ensemble methods achieve 92.08% and 91.25% accuracy with increased computational cost
- XGBoost training time: 0.82s among top performers

---

## 4. Final Model: XGBoost

### 4.1 Hyperparameters

**Optimal Configuration:**

```python
{
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.01,
    'subsample': 0.9,
    'colsample_bytree': 1.0,
    'scale_pos_weight': 3.0,
    'eval_metric': 'mlogloss'
}
```

### 4.2 Performance Evaluation

**1. Cross-Validation (5-Fold Stratified):**
- Mean F1-Score: 93.32%
- Standard Deviation: 0.39% (indicating high consistency)

**2. Test Set Metrics:**
- **Overall Accuracy:** 92.50%
- **Weighted F1-Score:** 92.40%
- **Macro ROC AUC:** 0.976

**Class-Wise Performance:**

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Rating 0 (Good) | 82% | 85% | 84% | 39 |
| Rating 1 (Excellent) | 94% | 97% | 95% | 175 |
| Rating 2 (Outstanding) | 100% | 77% | 87% | 26 |

**3. Confusion Matrix:**

```
Actual →       0    1    2
Predicted ↓
    0         33    6    0
    1          6  169    0
    2          1    5   20
```

**Misclassification Analysis:**
- Misclassifications primarily occur between adjacent classes
- 6 Good (Rating 0) employees misclassified as Excellent (Rating 1)
- 6 Excellent (Rating 1) employees misclassified as Good (Rating 0)
- 5 Excellent (Rating 1) employees misclassified as Outstanding (Rating 2)
- 1 Outstanding (Rating 2) employee misclassified as Good (Rating 0)

---

## 5. Feature Importance Analysis

**SHAP Values Comparison Across Models:**

| Feature | XGBoost | Random Forest | LightGBM | CatBoost |
|---------|---------|---------------|----------|----------|
| EmpEnvironmentSatisfaction | 0.4945 | 0.2712 | 1.2061 | 1.2638 |
| EmpLastSalaryHikePercent | 0.3720 | 0.2990 | 1.0529 | 1.3524 |
| YearsSinceLastPromotion | 0.2631 | 0.1229 | 0.8972 | 0.8344 |
| EmpWorkLifeBalance | 0.1310 | 0.0549 | 0.3490 | 0.2972 |
| EmpDepartment_Development | 0.1015 | 0.0412 | 0.3516 | 0.3827 |

**Analysis:**

1. **Cross-Model Feature Importance:**
   - EmpEnvironmentSatisfaction and EmpLastSalaryHikePercent rank in top 2 positions across all models
   - These two features show consistent predictive power

2. **Model-Specific Patterns:**
   - Tree-based models (LightGBM, CatBoost) assign higher importance to YearsSinceLastPromotion
   - XGBoost shows more balanced importance distribution across features

3. **Secondary Features:**
   - EmpJobRole_freq and TotalWorkExperienceInYears show lower but consistent importance
   - Work-life balance and department affiliation demonstrate moderate predictive power

---

## 6. Discussion

**1. Class Imbalance Handling:**
- Class weights improved minority class (Rating 2) recall from 77% to 87% compared to baseline models
- Rating 1 (majority class) predictions balanced with F1-score of 95%
- Weighted approach reduced model bias toward majority class

**2. Overfitting Control:**
- XGBoost's shallow trees (max_depth=3) minimized overfitting
- Overfit score of 0.0083 (0.83% train-test gap)
- Low learning rate (0.01) with high n_estimators (300) for stable convergence

**3. Ensemble Performance:**
- Voting/Stacking ensembles achieved 92.08% and 91.25% accuracy with increased computational cost
- Single XGBoost model achieved highest accuracy with faster training time
- Ensemble complexity added minimal performance gain

**4. Limitations:**
- Rating 2 (Outstanding) had lower recall (77%) due to limited samples (26 test instances, 11% of test set)
- KNN and Naive Bayes struggled with non-linear decision boundaries
- Model performance may degrade with distribution shifts in production

---

## 7. Recommendations

**1. Deployment Strategy:**
- Use XGBoost for real-time predictions due to speed (0.82s training), accuracy (92.5%), and minimal overfitting
- Implement REST API wrapper for production integration
- Set prediction confidence threshold at 80% for high-stakes decisions

**2. Monitoring and Maintenance:**
- Track class distribution shifts in production data
- Recalibrate class weights if distribution changes significantly
- Retrain model quarterly with new employee data
- Monitor prediction accuracy on holdout validation set

**3. Feature Engineering:**
- Test interaction features between EmpEnvironmentSatisfaction and YearsSinceLastPromotion
- Add temporal features if longitudinal data becomes available
- Analyze department-specific satisfaction patterns

**4. Data Collection:**
- Increase sample size for Rating 2 (Outstanding) class to improve recall
- Collect additional features: manager performance scores, team dynamics, project complexity
- Test model fairness across demographic subgroups (age, gender, education)

**5. System Integration:**
- Deploy model for performance management predictions
- Integrate with HRIS for automated feature extraction
- Build dashboard for HR team with prediction explanations (SHAP values)

---

## 8. Conclusion

The XGBoost model achieved 92.5% accuracy with minimal overfitting (0.83% train-test gap) for predicting employee performance ratings. Cross-validation demonstrates stable performance (93.32% ± 0.39%).

**Model Performance:**
- Test accuracy: 92.5% with minimal overfitting (0.0083)
- Cross-validation: 93.32% ± 0.39%
- Feature analysis: SHAP values identify key predictors
- Training time: 0.82s suitable for frequent retraining

**Application:**
- Performance prediction for current employees
- Data-driven performance management decisions
- Interpretable predictions using SHAP explanations
- Resource allocation based on identified impact factors

**Future Work:**
- Expand dataset for underrepresented classes (Rating 2: Outstanding)
- Test model fairness across employee subgroups
- Add temporal dynamics using longitudinal data
- Deploy production pipeline with automated monitoring

The model provides data-driven employee performance predictions for INX Future Inc. with training and deployment pipelines for ongoing use.

---

## Appendix

**Model Persistence:**
- Trained model saved as: `models/XGBClassifier_model_multiclass.pkl`
- Model size: Approximately 150 KB
- Loading time: < 0.1 seconds

**Reproducibility:**
- Random seed: 42 (used across all experiments)
- Training environment: Python 3.8+, XGBoost 2.0, scikit-learn 1.3
- Complete code available in project notebooks

**Performance Metrics Definition:**
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
- ROC AUC: Area under ROC curve (One-vs-Rest for multiclass)

---

**End of Report**
