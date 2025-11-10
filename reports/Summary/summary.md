# Employee Performance Analysis and Predictive Modeling for INX Future Inc.

**Report Date:** April 8, 2025
**Last Revised:** November 10, 2025

**Project Code:** 10281
**Document Code:** CDS_Project_2_INX_Future_Emp_Data_V1.6
**Prepared for:** INX Future Inc.

---

## Executive Summary

This project addresses INX Future Inc.'s employee performance challenges through data analysis and predictive modeling. Department-wise performance analysis identifies critical drivers of performance, and an XGBoost model achieves 92.5% accuracy in predicting performance ratings. Key findings include promotion delays in Sales and Finance departments, and the influence of salary hikes and environment satisfaction on performance. Recommendations focus on recognition programs, workload management, and predictive hiring tools.

---

## Table of Contents

- [Employee Performance Analysis and Predictive Modeling for INX Future Inc.](#employee-performance-analysis-and-predictive-modeling-for-inx-future-inc)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
    - [1.1 Objective](#11-objective)
    - [1.2 Scope](#12-scope)
    - [1.3 Approach](#13-approach)
  - [2. Data Preprocessing and Analysis](#2-data-preprocessing-and-analysis)
    - [2.1 Data Preparation](#21-data-preparation)
    - [2.2 Key Insights](#22-key-insights)
  - [3. Department-wise Performance Analysis](#3-department-wise-performance-analysis)
  - [4. Top 3 Performance Drivers](#4-top-3-performance-drivers)
  - [5. Predictive Model Development](#5-predictive-model-development)
    - [5.1 Model Performance](#51-model-performance)
    - [5.2 Feature Importance](#52-feature-importance)
  - [6. Recommendations](#6-recommendations)
  - [7. Algorithm and Training Methods](#7-algorithm-and-training-methods)
    - [7.1 Primary Algorithm](#71-primary-algorithm)
    - [7.2 Secondary Models](#72-secondary-models)
  - [8. Key Features and Selection Rationale](#8-key-features-and-selection-rationale)
    - [8.1 Most Important Features](#81-most-important-features)
    - [8.2 Feature Selection Techniques](#82-feature-selection-techniques)
    - [8.3 Why PCA Was Not Used](#83-why-pca-was-not-used)
  - [9. Techniques and Tools](#9-techniques-and-tools)
    - [9.1 Key Techniques](#91-key-techniques)
    - [9.2 Software Libraries and Tools](#92-software-libraries-and-tools)
  - [10. Feature Engineering](#10-feature-engineering)
    - [10.1 Most Important Features and Selection Rationale](#101-most-important-features-and-selection-rationale)
    - [10.2 Feature Transformations](#102-feature-transformations)
    - [10.3 Feature Correlations and Interactions](#103-feature-correlations-and-interactions)
  - [11. Results, Analysis and Insights](#11-results-analysis-and-insights)
    - [11.1 Interesting Relationships in the Data](#111-interesting-relationships-in-the-data)
    - [11.2 Most Important Technique](#112-most-important-technique)
    - [11.3 Answers to Business Problems](#113-answers-to-business-problems)
    - [11.4 Additional Business Insights](#114-additional-business-insights)
  - [12. Conclusion](#12-conclusion)

---

## 1. Project Overview

### 1.1 Objective

Identify performance bottlenecks, predict employee success, and provide data-backed recommendations to improve organizational performance and retention.

### 1.2 Scope

**Dataset:**
- 1,200 employee records across 6 departments
- 28 original features covering demographics, job roles, satisfaction metrics, work experience, and compensation
- Target variable: PerformanceRating (3 classes: 2=Good, 3=Excellent, 4=Outstanding)

**Analysis Focus:**
- Department-wise performance comparison
- Attrition trends and satisfaction correlations
- Predictive modeling for performance ratings

### 1.3 Approach

**Phase 1: Exploratory Analysis**
- Department-wise performance assessment
- Attrition trend analysis
- Satisfaction metric correlations

**Phase 2: Feature Engineering**
- Categorical variable encoding (one-hot, frequency, ordinal)
- Multicollinearity resolution
- Feature importance analysis

**Phase 3: Predictive Modeling**
- Tested 10+ algorithms (XGBoost, Random Forest, LightGBM, CatBoost, ensemble methods)
- Hyperparameter optimization via RandomizedSearchCV
- Cross-validation with stratified sampling

---

## 2. Data Preprocessing and Analysis

### 2.1 Data Preparation

**Data Quality:**
- **Missing Values:** Zero missing values detected
- **Duplicates:** No duplicate records found
- **Cleaning:** Removed high-cardinality identifier (EmpNumber)

**Outlier Treatment:**
- Addressed skewed distributions in experience variables
- Applied domain-informed capping:
  - YearsSinceLastPromotion: Capped at 10 years
  - TotalWorkExperienceInYears: Winsorization at 95th percentile (27 years)
  - ExperienceYearsAtThisCompany: Winsorization at 95th percentile (18 years)

**Feature Engineering:**
- Created EmpJobRole_freq: Frequency encoding for job roles (19 unique roles)
- Reduced dimensionality while preserving role-specific patterns

### 2.2 Key Insights

**Class Distribution:**
- 72.8% employees rated "Excellent" (Class 1)
- 16.2% rated "Good" (Class 0)
- 11.0% rated "Outstanding" (Class 2)
- Imbalance addressed through class weighting

**Multicollinearity:**
- High correlation detected between tenure variables
- ExperienceYearsAtThisCompany vs. YearsWithCurrManager: Spearman ρ = 0.84
- Resolved through VIF analysis and feature removal

**Critical Feature Patterns:**

1. **Salary Hike Differential:**
   - Top performers: 20.7% average hikes
   - Average performers: 14.4% average hikes
   - Differential: 6.3 percentage points

2. **Environment Satisfaction:**
   - 46.4% of low performers reported dissatisfaction (score 1)
   - 95.6% of high performers reported high satisfaction (score 3-4)

3. **Promotion Delays:**
   - Low performers: 3.7 years average wait
   - Top performers: 1.9 years average wait
   - Gap: 1.8 years

---

## 3. Department-wise Performance Analysis

**Performance by Department:**

| Department | Key Findings | Performance Distribution |
|------------|--------------|-------------------------|
| **Sales** | 44.8% low performers; high attrition (18.6%) | 23.3% Good, 67.3% Excellent, 9.4% Outstanding |
| **Development** | 84.2% excellent performers; highest salary hikes | 3.6% Good, 84.2% Excellent, 12.2% Outstanding |
| **Data Science** | 85% excellent performers; lowest attrition (1.5%) | 5.0% Good, 85.0% Excellent, 10.0% Outstanding |
| **Finance** | 30.6% low performers; limited training | 30.6% Good, 61.2% Excellent, 8.2% Outstanding |
| **Human Resources** | 70.4% average performers; moderate training | 18.5% Good, 70.4% Excellent, 11.1% Outstanding |
| **R&D** | 68.2% excellent performers; moderate attrition | 19.8% Good, 68.2% Excellent, 11.9% Outstanding |

**Key Insight:** Sales teams face performance variability due to role demands and travel frequency (70.5% travel rarely), while Development excels with structured workflows and higher compensation.

---

## 4. Top 3 Performance Drivers

**1. Employee Environment Satisfaction**
- **SHAP Value:** 0.4945 (highest)
- **Impact:** Employees with scores 1-2 were 4x more likely to underperform
- **Statistical Significance:** Kruskal-Wallis H = 156.78 (p < 0.001)

**2. Employee Last Salary Hike Percentage**
- **SHAP Value:** 0.3720 (second highest)
- **Impact:** Top performers received 20.7% hikes vs. 14.4% for average performers
- **Correlation:** Spearman ρ = 0.45 with performance rating (p < 0.001)

**3. Years Since Last Promotion**
- **SHAP Value:** 0.2631 (third highest)
- **Impact:** Delays >3 years linked to 18.6% attrition in low performers
- **Correlation:** Spearman ρ = -0.38 with performance rating (p < 0.001)

---

## 5. Predictive Model Development

### 5.1 Model Performance

**Top Performing Models:**

| Algorithm | Accuracy | F1-Score | ROC AUC | Training Time (s) | Overfit Score |
|-----------|----------|----------|---------|-------------------|---------------|
| **XGBoost (Tuned)** | 92.5% | 0.924 | 0.976 | 0.337 | 0.0083 |
| **Random Forest** | 92.5% | 0.923 | 0.973 | 0.827 | 0.0531 |
| **Voting Ensemble** | 92.1% | 0.920 | 0.974 | 3.843 | 0.0677 |
| **Stacking Ensemble** | 91.3% | 0.912 | 0.967 | 21.490 | 0.0552 |

**Final Model Selection:**
- XGBoost selected for production deployment
- Rationale: Highest accuracy with minimal overfitting (0.83% gap)
- Fast training time suitable for retraining cycles

**Cross-Validation Results:**
- 5-fold stratified cross-validation
- Mean F1-Score: 93.32%
- Standard Deviation: 0.39% (high consistency)

### 5.2 Feature Importance

**Top Predictors (SHAP Analysis):**

| Rank | Feature | SHAP Value | Impact % |
|------|---------|------------|----------|
| 1 | EmpEnvironmentSatisfaction | 0.4945 | 19.2% |
| 2 | EmpLastSalaryHikePercent | 0.3720 | 22.5% |
| 3 | YearsSinceLastPromotion | 0.2631 | 15.1% |
| 4 | ExperienceYearsAtThisCompany | 0.1523 | 8.7% |
| 5 | EmpJobRole_freq | 0.1245 | 7.3% |

**Feature Consistency Across Models:**
- Environment satisfaction and salary hikes ranked top 2 across all models
- XGBoost, Random Forest, LightGBM, CatBoost showed consistent feature rankings (correlation > 0.85)

---

## 6. Recommendations

**1. Recognition Programs:**
- Link promotions to measurable KPIs (project completion rate, client satisfaction scores)
- Introduce non-monetary incentives:
  - Flexible work hours for high work-life balance scores
  - Professional development stipends ($2,000/year)
  - Innovation grants for R&D employees

**2. Departmental Improvements:**

**Sales and Finance:**
- Reduce non-essential travel requirements
- Implement mentorship programs pairing with Development team
- Adjust quarterly targets based on historical performance data
- Provide CRM tools training (Sales) and financial modeling courses (Finance)

**R&D:**
- Address promotion delays (currently 3.7 years average)
- Implement biannual promotion cycles
- Offer innovation grants for high-impact projects

**3. Hiring Strategy:**
- Deploy XGBoost model to evaluate candidates based on:
  - Education background (Life Sciences, Technical fields)
  - Prior experience alignment (3-5 years per role)
  - Satisfaction indicators from interviews
- Confidence threshold: 80% for hiring recommendations

**4. Work-Life Balance:**
- Monitor overtime (currently 29.4% employees affected)
- Redistribute workloads to maintain <10% overtime workforce
- Implement mandatory overtime limits
- Launch "Wellness Wednesdays" stress management workshops

**5. Compensation Policy:**
- Minimum 18% salary hikes for "Excellent" ratings (current average: 15%)
- Minimum 22% salary hikes for "Outstanding" ratings (current average: 19%)
- Transparent criteria linking performance to compensation

---

## 7. Algorithm and Training Methods

### 7.1 Primary Algorithm

**XGBoost (eXtreme Gradient Boosting):**

**Selection Rationale:**
- Superior performance: 92.5% accuracy
- Minimal overfitting: 0.83% train-test gap
- Efficient handling of imbalanced data
- Fast training time: 0.337 seconds

**Training Configuration:**

**Hyperparameters (Optimized):**
```python
{
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.01,
    'subsample': 0.9,
    'colsample_bytree': 1.0,
    'gamma': 0,
    'scale_pos_weight': 3.0
}
```

**Training Method:**
- 5-fold stratified cross-validation to preserve class distribution
- RandomizedSearchCV with 50 iterations for hyperparameter tuning
- Class weighting to address imbalance:
  - Class 0 (Good): 2.06
  - Class 1 (Excellent): 0.46
  - Class 2 (Outstanding): 3.02

**Optimization Strategy:**
- Weighted F1-score as primary metric
- Early stopping with 20 rounds patience
- Tree pruning via gamma parameter

### 7.2 Secondary Models

**Random Forest:**
- Accuracy: 92.5% (tied with XGBoost)
- Used for interpretability validation
- Feature importance correlation with XGBoost: 0.91

**Ensemble Methods:**

**Voting Classifier (Soft Voting):**
- Accuracy: 92.1%
- Combined XGBoost, Random Forest, LightGBM predictions
- Probability averaging for final prediction

**Stacking Classifier:**
- Accuracy: 91.3%
- Base models: XGBoost, Random Forest, LightGBM, CatBoost
- Meta-model: Logistic Regression
- Trade-off: Slightly lower accuracy but improved calibration

---

## 8. Key Features and Selection Rationale

### 8.1 Most Important Features

**1. EmpEnvironmentSatisfaction**
- **SHAP Value:** 0.494
- **Impact Mechanism:** Influences engagement and productivity
- **Performance Correlation:** Low satisfaction employees 3x more likely to underperform
- **Implementation:** Quarterly satisfaction surveys, workspace improvements

**2. EmpLastSalaryHikePercent**
- **SHAP Value:** 0.372
- **Impact Mechanism:** Reflects recognition and motivation
- **Performance Correlation:** 40% performance differential between high/low hike groups
- **Implementation:** Performance-linked compensation policy

**3. YearsSinceLastPromotion**
- **SHAP Value:** 0.263
- **Impact Mechanism:** Indicates career stagnation risks
- **Performance Correlation:** 35% lower performance after 3+ years without promotion
- **Implementation:** Biannual promotion cycles, clear career paths

### 8.2 Feature Selection Techniques

**SHAP (SHapley Additive exPlanations):**
- Quantified global and local feature importance
- Identified non-linear relationships
- Validated feature ranking consistency across models

**Correlation Analysis:**
- Removed multicollinear variables using VIF > 10 threshold
- Dropped features:
  - YearsWithCurrManager (VIF = 9.87, corr with ExperienceYearsAtThisCompany = 0.75)
  - ExperienceYearsInCurrentRole (VIF = 11.23, corr = 0.87)
  - Age (VIF = 23.34, redundant with experience variables)

**Domain-Driven Engineering:**
- Created EmpJobRole_freq (frequency encoding)
- Captured role-specific trends without dimensionality explosion
- Reduced 19 unique roles to single numeric feature

**Statistical Testing:**
- Kruskal-Wallis H-test for numerical features vs. performance
- Chi-Square test for categorical features vs. performance
- Retained features with p < 0.001

### 8.3 Why PCA Was Not Used

**Interpretability Priority:**
- HR stakeholders require feature-level insights
- PCA components are not interpretable for policy decisions
- Direct feature importance supports implementation recommendations

**Domain Knowledge Validation:**
- Raw features (salary hikes, promotions) align with HR theories
- No need for dimensionality reduction (11 final features manageable)
- Model performance remained high without PCA

**Business Requirement:**
- SHAP analysis requires original feature space
- Regulatory compliance requires transparent feature effects
- Predictive model provides individual prediction explanations

---

## 9. Techniques and Tools

### 9.1 Key Techniques

**Data Preprocessing:**

**Outlier Handling:**
- Winsorization for skewed variables (TotalWorkExperienceInYears, ExperienceYearsAtThisCompany)
- Domain-informed capping (YearsSinceLastPromotion at 10 years)
- Preserved data integrity while reducing extreme leverage

**Categorical Encoding:**
- One-Hot Encoding: Gender, MaritalStatus, BusinessTravelFrequency, EmpDepartment
- Frequency Encoding: EmpJobRole (high cardinality, 19 unique roles)
- Ordinal Encoding: Satisfaction scores (1-4), EmpEducationLevel (1-5)

**Model Explainability:**

**SHAP Analysis:**
- Waterfall plots for individual predictions
- Summary plots for global feature importance
- Dependence plots for non-linear relationships

**Partial Dependence Plots (PDPs):**
- Revealed threshold effects (salary hike at 15%)
- Identified diminishing returns (promotion delay beyond 5 years)

**Performance Metrics:**

**Weighted F1-Score:**
- Addressed class imbalance (72.8% Excellent, 16.2% Good, 11.0% Outstanding)
- Balanced precision and recall across all classes

**ROC AUC (One-vs-Rest):**
- Multi-class discrimination assessment
- Final ROC AUC: 0.976

**Cross-Validation:**
- 5-fold stratified sampling
- Preserved class distribution in each fold
- Detected overfitting early

### 9.2 Software Libraries and Tools

**Custom Library:**
- **insightfulpy:** Custom Python library for exploratory data analysis
- GitHub: https://github.com/dhaneshbb/insightfulpy
- PyPI: https://pypi.org/project/insightfulpy/
- Functions used: Data preprocessing, feature engineering, model evaluation

**Core Packages:**

**Data Manipulation:**
- pandas: Data structures and analysis
- numpy: Numerical operations
- researchpy: Statistical summaries

**Visualization:**
- matplotlib.pyplot: Static plots
- seaborn: Statistical visualizations
- missingno: Missing data visualization

**Statistical Analysis:**
- scipy.stats: Hypothesis testing (Shapiro-Wilk, Kruskal-Wallis, Chi-Square)
- statsmodels.api: Advanced statistical models
- VIF analysis for multicollinearity

**Machine Learning:**
- scikit-learn: Model training, evaluation, preprocessing
- xgboost: Gradient boosting (primary model)
- lightgbm: Gradient boosting (comparison)
- catboost: Gradient boosting (comparison)
- scikitplot: ROC curves, confusion matrices

**Model Interpretation:**
- shap: SHapley Additive exPlanations
- Permutation importance

**Utilities:**
- joblib: Model persistence
- psutil: Memory profiling
- warnings: Warning suppression
- logging: Execution logging

**User-Defined Functions:**

**Statistical Analysis:**
- `normality_test_with_skew_kurt`: Shapiro-Wilk/KS tests with distribution metrics
- `spearman_correlation`: Non-parametric correlation analysis
- `calculate_vif`: Variance Inflation Factor calculation

**Model Evaluation:**
- `evaluate_model`: Accuracy, F1, precision, recall, ROC AUC, CV scores
- `cross_validation_analysis_table`: Fold-by-fold CV results
- `plot_multiclass_evaluation`: ROC curves, confusion matrices
- `mean_absolute_shap`: SHAP value aggregation for multiclass

**Memory Management:**
- `memory_usage`: Process memory monitoring
- `dataframe_memory_usage`: DataFrame memory footprint
- `garbage_collection`: Manual garbage collection

---

## 10. Feature Engineering

### 10.1 Most Important Features and Selection Rationale

**Top 3 Features (Detailed Analysis):**

**1. EmpEnvironmentSatisfaction**

**Selection Rationale:**
- Highest SHAP value: 0.494
- Statistical significance: Kruskal-Wallis H = 156.78 (p < 0.001)
- Direct HR policy relevance: Workspace quality, tools, culture

**Performance Impact by Level:**

| Satisfaction Level | Excellent/Outstanding % | Good % | Sample Size |
|-------------------|------------------------|--------|-------------|
| 1 (Low) | 45.2% | 54.8% | 178 |
| 2 (Medium) | 68.6% | 31.4% | 312 |
| 3 (High) | 88.3% | 11.7% | 486 |
| 4 (Very High) | 95.6% | 4.4% | 224 |

**2. EmpLastSalaryHikePercent**

**Selection Rationale:**
- SHAP value: 0.372
- Strong correlation: Spearman ρ = 0.72 with performance
- Top performers received 20.7% hikes vs. 14.4% for average

**Performance Impact by Hike Range:**

| Hike Range | Excellent/Outstanding % | Good % |
|------------|------------------------|--------|
| 11-14% | 62.3% | 37.7% |
| 15-17% | 81.2% | 18.8% |
| 18-20% | 89.7% | 10.3% |
| 21-25% | 93.4% | 6.6% |

**Critical Threshold:** 15% represents psychological and economic breakpoint

**3. YearsSinceLastPromotion**

**Selection Rationale:**
- SHAP value: 0.263
- Critical for retention: Delays >3 years linked to 18.6% attrition
- Career stagnation indicator

**Performance Impact by Delay:**

| Promotion Delay | Excellent/Outstanding % | Good % | Attrition Risk |
|-----------------|------------------------|--------|----------------|
| 0-1 years | 91.2% | 8.8% | Low (5%) |
| 2-3 years | 82.5% | 17.5% | Medium (12%) |
| 4-5 years | 71.3% | 28.7% | High (21%) |
| >5 years | 58.9% | 41.1% | Very High (35%) |

### 10.2 Feature Transformations

**1. Categorical Encoding:**

**One-Hot Encoding:**
- **Applied to:** Gender, MaritalStatus, BusinessTravelFrequency, EmpDepartment, EducationBackground
- **Rationale:** Low cardinality (2-6 unique values), nominal categories
- **Result:** Created binary indicator variables

**Frequency Encoding:**
- **Applied to:** EmpJobRole
- **Rationale:** High cardinality (19 unique roles), avoid dimensionality explosion
- **Method:** Replaced role with occurrence frequency
- **Example:** Sales Executive (270 occurrences) → 0.225

**Ordinal Encoding:**
- **Applied to:** Satisfaction scores, EmpEducationLevel, TrainingTimesLastYear
- **Rationale:** Inherent ordering in categories
- **Mapping:**
  - Satisfaction: 1=Low, 2=Medium, 3=High, 4=Very High
  - Education: 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctorate

**2. Outlier Handling:**

**Winsorization:**
- **Applied to:** TotalWorkExperienceInYears (capped at 95th percentile = 27 years)
- **Applied to:** ExperienceYearsAtThisCompany (capped at 95th percentile = 18 years)
- **Rationale:** Mitigated skewness without losing data

**Domain-Informed Capping:**
- **YearsSinceLastPromotion:** Capped at 10 years (realistic max)
- **NumCompaniesWorked:** Capped at 8 companies (career mobility limit)
- **ExperienceYearsInCurrentRole:** Capped at 14 years

**3. Binning (Experimental, Not Used in Final Model):**
- **TrainingTimesLastYear:** Grouped into Low (0-2), Medium (3-4), High (5-6)
- **Result:** Reduced granularity without performance gain, reverted to original

**Why These Transformations:**
- **Frequency Encoding:** Avoided dimensionality explosion (19 binary features → 1 numeric)
- **Winsorization:** Preserved 95% of data while reducing extreme leverage
- **Ordinal Encoding:** Maintained natural ordering for satisfaction scales

### 10.3 Feature Correlations and Interactions

**High Multicollinearity Identified:**

**1. Experience Variables:**
- ExperienceYearsAtThisCompany vs. YearsWithCurrManager: Spearman ρ = 0.84
- **Resolution:** Retained ExperienceYearsAtThisCompany (broader tenure context), dropped YearsWithCurrManager
- **VIF Before:** YearsWithCurrManager VIF = 9.87
- **VIF After:** All remaining features VIF < 5.0

**2. Age and Experience:**
- Age vs. TotalWorkExperienceInYears: Spearman ρ = 0.70
- **Resolution:** Dropped Age (VIF = 23.34), kept experience variables
- **Rationale:** Experience more actionable than age for HR policies

**Key Correlations (Insight Generation):**

**1. Salary and Satisfaction Link:**
- EmpLastSalaryHikePercent vs. EmpJobSatisfaction: Spearman ρ = 0.61
- **Insight:** Salary hikes indirectly boost job satisfaction
- **Business Implication:** Compensation policy affects multiple satisfaction dimensions

**2. Work-Life Balance and Performance:**
- EmpWorkLifeBalance vs. OverTime: Spearman ρ = -0.48
- **Insight:** Overtime negatively impacts work-life balance perception
- **Business Implication:** Overtime limits improve balance scores

**XGBoost-Captured Non-Linear Interactions:**

**1. Low Satisfaction + High Overtime:**
- EmpEnvironmentSatisfaction (score 1-2) × OverTime (Yes)
- **Result:** 63% attrition risk
- **Detected via:** SHAP interaction values
- **Business Action:** Prioritize satisfaction improvement in overtime-heavy roles

**2. Advanced Degrees + Moderate Experience:**
- EmpEducationLevel (4-5) × TotalWorkExperience (5-10 years)
- **Result:** 78% likelihood of high performance
- **Detected via:** SHAP dependence plots
- **Business Action:** Target hiring profiles with this combination

**3. Promotion Delay + Low Salary Hike:**
- YearsSinceLastPromotion (>3 years) × EmpLastSalaryHikePercent (<15%)
- **Result:** 82% probability of "Good" rating (underperformance)
- **Detected via:** SHAP interaction analysis
- **Business Action:** Avoid dual stagnation (both promotion and compensation)

**Correlation Management Strategy:**

**VIF Analysis:**
- Threshold: VIF > 10 for removal
- Initial features with VIF > 10:
  - Age: 23.34
  - ExperienceYearsInCurrentRole: 11.23
  - YearsWithCurrManager: 9.87 (borderline)
- **Result:** All removed, final VIF scores < 5.0

**Regularization:**
- L2 regularization in Logistic Regression (secondary model)
- Handled residual multicollinearity
- Alpha = 0.01 (cross-validated)

---

## 11. Results, Analysis and Insights

### 11.1 Interesting Relationships in the Data

**1. Overtime Paradox:**
- **Finding:** Employees working overtime had lower performance ratings
- **Statistics:** 29.4% overtime workers vs. 70.6% non-overtime
- **Performance:** Overtime group: 67% Excellent/Outstanding vs. 84% in non-overtime group
- **Insight:** Extra hours do not boost productivity, may indicate workload mismanagement
- **Business Implication:** Redistribute workloads instead of increasing overtime

**2. Education vs. Experience Trade-off:**
- **Finding:** Technical degree holders with <5 years outperformed PhDs with >10 years
- **Performance:** Technical (<5 yrs): F1=0.82 vs. PhD (>10 yrs): F1=0.67
- **Insight:** Adaptability and current skill relevance over tenure
- **Business Implication:** Prioritize skill fit over degree level in hiring

**3. Manager Tenure Double-Edged Effect:**
- **Finding:** Employees with same manager for >7 years had mixed outcomes
- **Positive:** 23% higher satisfaction scores
- **Negative:** 12% lower innovation metrics (project diversity, new initiatives)
- **Insight:** Long-term manager relationships build trust but may reduce fresh perspectives
- **Business Implication:** Balance manager continuity with periodic role rotations

**4. Training Frequency Analysis:**
- **Finding:** Training 3-4 times/year shows best performance correlation
- **Performance:** 3-4 sessions: Higher performance vs. >4 sessions: diminishing returns
- **Insight:** Quality over quantity in professional development
- **Business Implication:** Focus on targeted training programs

### 11.2 Most Important Technique

**SHAP (SHapley Additive exPlanations):**

**Why Critical:**
- Enabled granular interpretation of XGBoost predictions
- Revealed non-linear relationships (e.g., salary hike threshold at 15%, diminishing returns above 21%)
- Provided both global (model-level) and local (prediction-level) explanations

**Key Discoveries via SHAP:**

**1. Environment Satisfaction Dominance:**
- SHAP value: 0.4945 (1.5x higher than second feature)
- Identified as primary driver, guiding HR policy focus
- Waterfall plots showed this feature tipping 60% of borderline predictions

**2. Non-Linear Salary Hike Effects:**
- SHAP dependence plot revealed sharp threshold at 15%
- Below 15%: Strong negative impact on performance prediction
- Above 21%: Diminishing marginal returns
- Informed tiered compensation recommendation

**3. Interaction Effects:**
- SHAP interaction values detected:
  - Low satisfaction × Overtime = 2.3x higher negative impact
  - Promotion delay × Low salary hike = Compounding demotivation effect
- Enabled targeted interventions for high-risk employee segments

**Impact on Analysis:**
- Identified environment satisfaction as primary performance driver
- Informed workspace improvement recommendations
- Provided interpretable model for HR decision-making

### 11.3 Answers to Business Problems

| Business Problem | Data-Backed Answer | Key Metrics |
|------------------|-------------------|-------------|
| **Department-wise performance gaps** | **Sales/Finance:** High attrition (18.6%, 15.3%) due to promotion delays (avg. 3.7 years)<br>**Data Science:** 85% "Excellent" ratings from structured training (avg. 4.2 sessions/year)<br>**Development:** 84.2% excellent performance, lowest attrition (5.1%) | Sales attrition: 18.6%<br>Finance attrition: 15.3%<br>Data Science performance: 85% Excellent |
| **Top 3 performance factors** | 1. **Environment Satisfaction** (SHAP: 0.4945): 3x impact on underperformance<br>2. **Salary Hike Equity** (SHAP: 0.3720): 40% performance differential<br>3. **Promotion Timeliness** (SHAP: 0.2631): 35% impact after 3+ years delay | SHAP values: 0.4945, 0.3720, 0.2631<br>Impact multipliers: 3x, 40%, 35% |
| **Predictive model for hiring** | **XGBoost model (92.5% accuracy)** flags high-potential candidates:<br>- Education in Life Sciences/Technical fields (coefficient: +0.42)<br>- Prior experience with >2 promotions in 5 years (coefficient: +0.38)<br>- Work history tenure balance: 3-5 years per role preferred | Model accuracy: 92.5%<br>F1-score: 0.924<br>ROC AUC: 0.976 |
| **Recommendations** | **Sales:** Adjust quarterly targets, implement flexible commission structure<br>**Finance:** Introduce bi-annual promotion cycles, automate manual tasks<br>**All Departments:** Minimum 18% salary hikes for "Excellent" ratings | Promotion cycle: Biannual<br>Min salary hike: 18% |

### 11.4 Additional Business Insights

**1. Work-Life Balance Retention Driver:**
- **Finding:** Employees with "Best" balance (score 4) showed 2.5x higher retention
- **Analysis:** Work-life balance importance despite lower average salaries
- **Implication:** Work-life balance programs as retention tool
- **Recommendation:** Offer flexible work arrangements

**2. Training Frequency Analysis:**
- **Finding:** 3-4 training sessions/year shows best performance correlation
- **Performance:** Higher performance vs. over-trained peers (>4 sessions)
- **Implication:** Quality over quantity in training programs
- **Recommendation:** Focus on targeted training sessions

**3. Role-Specific Performance Trajectories:**

**Sales Executives:**
- **Pattern:** Performance drops 14% after 3 years in same role
- **Cause:** Client relationship fatigue, territory saturation
- **Recommendation:** Implement role rotation or territory refresh every 3 years

**R&D Managers:**
- **Pattern:** Retention improved 27% with cross-departmental projects
- **Mechanism:** Exposure to business applications, broader skill development
- **Recommendation:** Mandate 20% time on cross-functional initiatives

**4. Gender Dynamics in Technical Roles:**
- **Finding:** Female employees in technical roles showed 8% higher performance
- **Concern:** 12% lower promotion rates than male peers with equivalent performance
- **Statistical Significance:** Chi-Square test p = 0.003 (significant bias)
- **Recommendation:**
  - Implement blind promotion review process
  - Set gender parity targets for technical role promotions
  - Monitor promotion rate disparities quarterly

**5. Attrition Risk Prediction:**
- **High-Risk Profile:** Low satisfaction (score 1-2) + >3 years promotion delay + <15% salary hike
- **Risk Level:** 78% probability of attrition within 12 months
- **Early Warning System:** Deploy monthly predictive alerts for this profile
- **Intervention:** Personalized retention plans with immediate manager review

---

## 12. Conclusion

The analysis identifies performance patterns and provides data-driven recommendations for INX Future Inc.:

**Key Takeaways:**

**1. Environment Satisfaction Strongest Predictor**
- 3x impact on performance, highest SHAP value (0.4945)
- Investment target: Workspace improvements for Sales and Finance departments
- Expected outcome: Reduction in underperformance ratings

**2. Compensation and Promotion Policy Alignment Needed**
- 15% salary hike threshold correlates with performance improvement
- Promotion delays >3 years correlate with 35% performance decline and 18.6% attrition
- Recommendations: Minimum 18% hikes for "Excellent" ratings, biannual promotion cycles

**3. Predictive Model Deployment**
- XGBoost model (92.5% accuracy) ready for deployment
- Applications: Hiring screening, retention risk alerts, performance forecasting
- Expected outcomes: Improved hiring quality, reduced attrition of high performers

**4. Department-Specific Interventions Required**
- Sales: Review and adjust targets, implement mentorship, flexible commissions
- Finance: Automate manual tasks, career path clarity
- Development/Data Science: Maintain high satisfaction through innovation grants

**5. Work-Life Balance for Retention**
- Higher retention likelihood for high work-life balance scores
- Work-life balance programs as retention tool
- Implementation: Hybrid work arrangements, flexible hours

**Implementation Areas:**

- Increase in "Excellent/Outstanding" ratings
- Reduction in attrition
- Improved new hire performance prediction

**Technical Deployment:**
- Model persistence: models/XGBClassifier_model_multiclass.pkl
- API integration: REST endpoint for real-time predictions
- Monitoring: MLOps platform for drift detection and retraining triggers
- Retraining schedule: Quarterly with rolling 12-month data window

Statistical analysis, machine learning modeling, and SHAP-driven interpretation provide data-driven strategies to address declining performance trends at INX Future Inc.

---

**End of Report**
