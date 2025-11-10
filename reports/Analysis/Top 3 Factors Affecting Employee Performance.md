# Top 3 Factors Affecting Employee Performance

**Report Date:** April 8, 2025
**Last Revised:** November 10, 2025

**Project Code:** 10281
**Document Code:** CDS_Project_2_INX_Future_Emp_Data_V1.6
**Prepared for:** INX Future Inc.

---

## Executive Summary

Using machine learning models (including XGBoost, Random Forest, and ensemble methods) trained on employee performance data from 1,200 employees, the following three factors show the strongest influence on employee performance at INX Future Inc.:

1. **Employee Environment Satisfaction**
2. **Employee Last Salary Hike Percentage**
3. **Years Since Last Promotion**

These factors were identified through feature importance analysis using SHAP (SHapley Additive exPlanations) values and model interpretability techniques. The XGBoost model achieved 92.5% accuracy with 92.4% F1-score on test data. This report provides a detailed breakdown of their impact and implementation recommendations.

---

## Table of Contents

- [Top 3 Factors Affecting Employee Performance](#top-3-factors-affecting-employee-performance)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [1. Methodology](#1-methodology)
  - [2. Factor 1: Employee Environment Satisfaction](#2-factor-1-employee-environment-satisfaction)
    - [2.1 Impact Analysis](#21-impact-analysis)
    - [2.2 Statistical Evidence](#22-statistical-evidence)
    - [2.3 Implementation](#23-implementation)
  - [3. Factor 2: Employee Last Salary Hike Percentage](#3-factor-2-employee-last-salary-hike-percentage)
    - [3.1 Impact Analysis](#31-impact-analysis)
    - [3.2 Statistical Evidence](#32-statistical-evidence)
    - [3.3 Implementation](#33-implementation)
  - [4. Factor 3: Years Since Last Promotion](#4-factor-3-years-since-last-promotion)
    - [4.1 Impact Analysis](#41-impact-analysis)
    - [4.2 Statistical Evidence](#42-statistical-evidence)
    - [4.3 Implementation](#43-implementation)
  - [5. Model Validation and Supporting Evidence](#5-model-validation-and-supporting-evidence)
  - [6. Recommendations](#6-recommendations)
  - [7. Conclusion](#7-conclusion)

---

## 1. Methodology

**Analysis Approach:**
- Machine learning models trained on 1,200 employee records with 11 features
- Multiple algorithms evaluated: XGBoost, Random Forest, LightGBM, CatBoost, ensemble methods
- Feature importance measured using SHAP (SHapley Additive exPlanations) values
- Cross-validation with 5-fold stratified sampling
- Statistical hypothesis testing (Kruskal-Wallis, Chi-Square)

**Model Selection:**
- XGBoost selected as final model based on 92.5% accuracy and minimal overfitting (0.83%)
- SHAP values used for feature importance ranking
- Consistent findings across multiple models (Random Forest, LightGBM, CatBoost)

**Data Quality:**
- 1,200 employees across 6 departments
- 27 original features, reduced to 11 predictors after feature selection
- Zero missing values, no duplicates
- Stratified 80/20 train-test split

---

## 2. Factor 1: Employee Environment Satisfaction

### 2.1 Impact Analysis

**Ranking:** Most critical feature (Rank 1 of 11)

**Definition:**
- Employee satisfaction with work environment on scale 1-4
- Encompasses workspace quality, tools/resources, organizational culture, team dynamics

**Performance Impact:**
- Employees with low satisfaction (score 1-2) are **3x more likely to underperform**
- High satisfaction (score 3-4) correlates with 95.6% "Excellent/Outstanding" ratings
- Direct correlation with client satisfaction metrics

**Departmental Variation:**
- High-performing departments (Development, Data Science): 75-80% employees in high satisfaction levels
- Underperforming departments (Sales, Finance): 40-50% employees in low satisfaction levels

### 2.2 Statistical Evidence

**SHAP Value:** 0.4945 (highest across all features)

**Feature Importance Comparison:**

| Model | Environment Satisfaction Importance |
|-------|-------------------------------------|
| XGBoost | 0.4945 |
| Random Forest | 0.2712 |
| LightGBM | 1.2061 |
| CatBoost | 1.2638 |

**Statistical Testing:**
- Kruskal-Wallis H-statistic: 156.78 (p < 0.001)
- Chi-Square statistic: 89.34 (p < 0.001)
- Both tests confirm highly significant relationship with performance

**Performance by Satisfaction Level:**

| Satisfaction Level | Excellent/Outstanding % | Good % |
|-------------------|------------------------|--------|
| 1 (Low) | 45.2% | 54.8% |
| 2 (Medium) | 68.6% | 31.4% |
| 3 (High) | 88.3% | 11.7% |
| 4 (Very High) | 95.6% | 4.4% |

### 2.3 Implementation

**Actions:**

1. **Quarterly Workplace Satisfaction Surveys:**
   - Anonymous submission with department-level reporting
   - Focus areas: Manager communication, resource availability, workspace quality
   - Action plans developed within 2 weeks of survey completion

2. **Workspace Improvements:**
   - Priority: Sales and Finance departments (40-50% low satisfaction)
   - Ergonomic upgrades, technology infrastructure modernization
   - Collaborative spaces for team projects

3. **Team-Building Activities:**
   - Monthly team events to strengthen culture
   - Cross-departmental collaboration initiatives
   - Recognition programs for high performers

**Expected Outcomes:**
- Increase in high satisfaction scores
- Reduction in underperformance ratings
- Improvement in client satisfaction metrics

---

## 3. Factor 2: Employee Last Salary Hike Percentage

### 3.1 Impact Analysis

**Ranking:** Second most critical feature (Rank 2 of 11)

**Definition:**
- Percentage salary increase in most recent review cycle
- Range: 11-25% in dataset
- Reflects compensation competitiveness and recognition

**Performance Impact:**
- Employees with hikes below 15% show **40% lower performance ratings**
- Hikes above 20% correlate with 93.4% "Excellent/Outstanding" ratings
- Strong linkage to retention: Low hikes correlate with 25% higher turnover

**Threshold Effect:**
- Sharp performance improvement at 15% threshold
- Minimal additional performance gain above 21%

### 3.2 Statistical Evidence

**SHAP Value:** 0.3720 (second highest)

**Feature Importance Comparison:**

| Model | Salary Hike Importance |
|-------|------------------------|
| XGBoost | 0.3720 |
| Random Forest | 0.2990 |
| LightGBM | 1.0529 |
| CatBoost | 1.3524 |

**Statistical Testing:**
- Kruskal-Wallis H-statistic: 89.45 (p < 0.001)
- Spearman correlation with target: 0.45 (p < 0.001)

**Performance by Hike Range:**

| Hike Range | Excellent/Outstanding % | Good % |
|------------|------------------------|--------|
| 11-14% | 62.3% | 37.7% |
| 15-17% | 81.2% | 18.8% |
| 18-20% | 89.7% | 10.3% |
| 21-25% | 93.4% | 6.6% |

**Key Finding:** 15% represents critical psychological and economic breakpoint

### 3.3 Implementation

**Compensation Strategy:**

1. **Performance-Linked Salary Hikes:**
   - "Excellent" ratings: Minimum 18% salary hikes (current average: 15%)
   - "Outstanding" ratings: Minimum 22% salary hikes (current average: 19%)
   - Annual review cycle with mid-year adjustments for high performers

2. **Transparent Compensation Policies:**
   - Publish clear criteria linking performance ratings to hike percentages
   - Benchmark against industry standards annually
   - Communicate compensation philosophy to all employees

3. **Non-Monetary Incentives:**
   - Stock options for high performers
   - Recognition programs and awards
   - Professional development stipends
   - Flexible benefits packages

**Expected Outcomes:**
- Alignment of salary hikes with performance ratings
- Reduced compensation-related attrition
- Improved retention of high performers

---

## 4. Factor 3: Years Since Last Promotion

### 4.1 Impact Analysis

**Ranking:** Third most critical feature (Rank 3 of 11)

**Definition:**
- Time elapsed since employee's most recent promotion
- Range: 0-15 years (capped at 10 years for outlier treatment)
- Indicator of career progression and growth opportunities

**Performance Impact:**
- Employees unpromoted for 3+ years exhibit **35% lower performance**
- Recently promoted employees (0-1 years) show 91.2% "Excellent/Outstanding" ratings
- Career stagnation leads to disengagement and reduced motivation

**Departmental Patterns:**
- Development: Shorter promotion cycles (avg. 1.8 years), higher performance
- Sales: Longer delays (avg. 3.7 years), lower performance and higher attrition

### 4.2 Statistical Evidence

**SHAP Value:** 0.2631 (third highest)

**Feature Importance Comparison:**

| Model | Years Since Promotion Importance |
|-------|----------------------------------|
| XGBoost | 0.2631 |
| Random Forest | 0.1229 |
| LightGBM | 0.8972 |
| CatBoost | 0.8344 |

**Statistical Testing:**
- Kruskal-Wallis H-statistic: 67.23 (p < 0.001)
- Spearman correlation with target: -0.38 (p < 0.001, negative correlation)

**Performance by Promotion Delay:**

| Promotion Delay | Excellent/Outstanding % | Good % |
|-----------------|------------------------|--------|
| 0-1 years | 91.2% | 8.8% |
| 2-3 years | 82.5% | 17.5% |
| 4-5 years | 71.3% | 28.7% |
| >5 years | 58.9% | 41.1% |

**Career Stagnation Effect:**
- Steepest performance decline between years 2-4
- Non-linear relationship with diminishing negative impact after 5 years

### 4.3 Implementation

**Career Development Strategy:**

1. **Structured Promotion Cycles:**
   - Implement biannual promotion reviews
   - Maximum 2-year wait between promotions for high performers
   - Clear promotion criteria published for all job levels
   - Priority: Sales and R&D departments (average 3.7 years delay)

2. **Lateral Growth Opportunities:**
   - Cross-departmental project assignments
   - Leadership training programs
   - Technical skill certifications
   - Mentorship roles for senior employees

3. **Predictive Career Planning:**
   - Use analytics to identify high-potential employees
   - Individual development plans for all employees
   - Career path transparency and communication
   - Accelerated development programs for high performers

**Expected Outcomes:**
- Reduction in average promotion delay
- Increase in "Outstanding" ratings
- Improved retention of high performers

---

## 5. Model Validation and Supporting Evidence

**Final Model Performance:**

**XGBoost Classifier Metrics:**
- **Accuracy:** 92.5% (222 of 240 test samples correct)
- **F1-Score:** 92.4% (weighted average across classes)
- **ROC AUC:** 0.976 (One-vs-Rest for multiclass)
- **Overfit Score:** 0.0083 (0.83% train-test gap)

**Cross-Validation Consistency:**
- **Mean F1-Score:** 93.32%
- **Standard Deviation:** 0.39% (indicates high stability)
- 5-fold stratified cross-validation

**Class-Wise Performance:**

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Rating 2 (Good) | 82% | 85% | 84% | 39 |
| Rating 3 (Excellent) | 94% | 97% | 95% | 174 |
| Rating 4 (Outstanding) | 100% | 77% | 87% | 27 |

**Confusion Matrix Analysis:**
```
Actual →       2    3    4
Predicted ↓
    2         33    6    0
    3          6  169    5
    4          0    0   21
```

**Key Observations:**
- **Rating 3 (Excellent):** Best performance (97% recall), benefits from majority class
- **Rating 4 (Outstanding):** Perfect precision (100%) but lower recall (77%), indicating conservative prediction
- **Rating 2 (Good):** Balanced performance (85% recall)

**Model Robustness:**
- Minimal overfitting validates generalization capability
- Consistent feature importance across multiple algorithms
- High cross-validation stability confirms reliability

---

## 6. Recommendations

**1. Training Programs:**
- Address skill gaps in departments with low environment satisfaction
- CRM tools training for Sales department
- Financial modeling and analytics for Finance department
- Leadership development for managers

**2. Compensation Policies:**
- Link salary hikes directly to performance ratings
- Implement performance analytics dashboard
- Quarterly compensation reviews for high performers
- Benchmark against market rates annually

**3. Promotion Transparency:**
- Publish clear promotion criteria and timelines
- Communicate career paths for all job levels
- Quarterly career development workshops
- Individual development plans with promotion targets

**4. Predictive Hiring and Retention:**
- Use trained XGBoost model to evaluate candidate alignment with performance drivers
- Screen for environment satisfaction fit during interviews
- Assess salary expectations against performance potential
- Predictive attrition alerts for early intervention

**5. Department-Specific Interventions:**
- **Sales:** Review and adjust targets, provide mentorship, implement flexible commission structure
- **Finance:** Automate manual tasks, provide training, clarify career paths
- **Development:** Maintain high satisfaction through innovation grants and recognition

**6. Leadership Development:**
- Manager training on performance coaching and conflict resolution
- 360-degree feedback for all managers
- Team satisfaction scores tied to manager performance reviews

---

## 7. Conclusion

The three factors identified through machine learning analysis show the strongest influence on employee performance at INX Future Inc.:

1. **Environment Satisfaction** (SHAP: 0.4945) - Strongest predictor with 3x impact on performance
2. **Salary Hikes** (SHAP: 0.3720) - Second strongest with 40% performance differential
3. **Promotion Timing** (SHAP: 0.2631) - Third strongest with 35% impact on engagement

Interventions targeting these factors can address identified performance gaps.

**Implementation Areas:**
- Environment satisfaction improvement programs
- Compensation policy alignment with performance ratings
- Promotion cycle standardization
- Predictive model deployment for hiring decisions

**Model Deployment:**
- XGBoost model saved for production use
- 92.5% accuracy validated on test data
- Integration into HR analytics systems
- Quarterly retraining recommended

These findings, validated through statistical analysis and machine learning models, provide data-driven strategies to address declining performance trends at INX Future Inc.

---

**End of Report**
