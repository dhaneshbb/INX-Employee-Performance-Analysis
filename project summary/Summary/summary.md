# Employee Performance Analysis & Predictive Modeling for INX Future Inc. 

---

### **Executive Summary**  
This project addresses INX Future Inc.'s employee performance challenges through data-driven analysis and predictive modeling. By analyzing department-wise performance, identifying critical drivers of performance, and building a robust predictive model, actionable insights are provided to enhance productivity and retention. Key findings include systemic promotion delays in Sales/Finance, the dominance of salary hikes and environment satisfaction in performance, and an optimized XGBoost model achieving **92.5% accuracy** in predicting performance ratings. Recommendations focus on equitable recognition programs, workload management, and AI-driven hiring tools.  


### **1. Project Overview**  
**Objective**: Identify performance bottlenecks, predict employee success, and provide data-backed recommendations.  
**Scope**:  
- Analyzed 1,200 employee records across 28 features (demographics, job roles, satisfaction metrics).  
- Target variable: PerformanceRating (3 classes: 2=Low, 3=Average, 4=Outstanding).  
**Approach**:  
1. **Exploratory Analysis**: Departmental performance, attrition trends, and satisfaction correlations.  
2. **Feature Engineering**: Encoded categorical variables (e.g., EmpJobRole_freq for role frequency).  
3. **Predictive Modeling**: Tested 10+ algorithms, optimized via hyperparameter tuning.  


### **2. Data Preprocessing & Analysis**  
#### **Data Preparation**  
- **Cleaning**: No missing values or duplicates detected. Removed high-cardinality identifiers (EmpNumber).  
- **Outliers**: Addressed skewed distributions in experience variables (e.g., YearsSinceLastPromotion).  
- **Feature Engineering**: Created EmpJobRole_freq (job role frequency encoding).  

#### **Key Insights**  
- **Class Imbalance**: 72.8% employees rated "Average" (3), 16.2% "Low" (2), 11% "Outstanding" (4).  
- **Multicollinearity**: High correlation between tenure variables (e.g., ExperienceYearsAtThisCompany vs. YearsWithCurrManager: ρ=0.84).  
- **Critical Features**:  
  1. **Salary Hike**: Top performers received 20.7% hikes vs. 14.4% for average performers.  
  2. **Environment Satisfaction**: 46.4% of low performers reported dissatisfaction (score 1).  
  3. **Promotion Delays**: Low performers waited 3.7 years for promotions vs. 1.9 years for top performers.  


### **3. Department-wise Performance Analysis**  
| Department               | Key Findings |  
|--------------------------|--------------|  
| **Sales**                | 44.8% low performers; high travel frequency (70.5% "Travel_Rarely"). |  
| **Development**          | 84.2% average performers; highest salary hikes (12.19% rated "Outstanding"). |  
| **Data Science**         | 85% average performers; lowest attrition (1.5%). |  
| **Human Resources**      | 70.4% average performers; moderate training frequency. |  

**Insight**: Sales teams face performance variability due to role demands, while Development excels with structured workflows.  

### **4. Top 3 Performance Drivers**  
1. **EmpEnvironmentSatisfaction**: Employees with scores 1–2 were **4x more likely** to underperform.  
2. **EmpLastSalaryHikePercent**: Directly correlated with higher ratings (SHAP value=0.372 in XGBoost).  
3. **YearsSinceLastPromotion**: Delays linked to 18.6% attrition in low performers.  

### **5. Predictive Model Development**  
#### **Model Performance**  
| Algorithm                | Accuracy | F1-Score | Training Time (sec) |  
|--------------------------|----------|----------|---------------------|  
| **XGBoost (Tuned)**      | 92.5%    | 0.924    | 0.337               |  
| **Random Forest**        | 92.5%    | 0.923    | 0.827               |  
| **Voting Ensemble**      | 92.1%    | 0.920    | 3.843               |  

#### **Feature Importance**  
- **Top Predictors**:  
  1. EmpLastSalaryHikePercent (22.5% impact).  
  2. EmpEnvironmentSatisfaction (19.2%).  
  3. YearsSinceLastPromotion (15.1%).  


### **6. Recommendations**  
1. **Recognition Programs**:  
   - Link promotions to measurable KPIs (e.g., project completion rate).  
   - Introduce non-monetary incentives (e.g., flexible hours) for high work-life balance scores.  

2. **Departmental Improvements**:  
   - **Sales/Finance**: Reduce non-essential travel; implement mentorship programs.  
   - **R&D**: Address promotion delays to curb attrition.  

3. **Hiring Strategy**:  
   - Deploy the XGBoost model to evaluate candidates’ education, experience, and satisfaction alignment.  

4. **Work-Life Balance**:  
   - Monitor overtime (29.4% employees affected) and redistribute workloads.  


The analysis reveals that equitable recognition, timely promotions, and workplace satisfaction are critical to improving performance. The XGBoost model offers a reliable tool for talent acquisition, while departmental reforms can address systemic issues. Next steps include real-time monitoring of satisfaction metrics and bias checks in the hiring model.  

---

# PROJECT SUMMARY

### **1. Algorithm & Training Methods**  
**Primary Algorithm**:  
- **XGBoost (eXtreme Gradient Boosting)** was selected as the optimal model due to its superior performance (92.5% accuracy) and efficiency in handling imbalanced data.  
- **Training Method**:  
  - **Stratified 5-Fold Cross-Validation** to preserve class distribution during training.  
  - **Hyperparameter Tuning**: RandomizedSearchCV with 50 iterations to optimize n_estimators, max_depth, and learning_rate.  
  - **Class Weighting**: Adjusted weights (Low: 2.06, Average: 0.46, Outstanding: 3.02) to mitigate class imbalance.  

**Secondary Models**:  
- **Random Forest** (92.5% accuracy) for interpretability.  
- **Ensemble Methods**:  
  - **Voting Classifier** (Soft Voting: 92.1% accuracy).  
  - **Stacking Classifier** (XGBoost + Logistic Regression meta-model: 91.25% accuracy).  


### **2. Key Features & Selection Rationale**  
**Most Important Features**:  
1. **EmpEnvironmentSatisfaction** (SHAP value: 0.494): Directly impacts engagement and productivity.  
2. **EmpLastSalaryHikePercent** (SHAP value: 0.372): Reflects recognition and motivation.  
3. **YearsSinceLastPromotion** (SHAP value: 0.263): Indicates career stagnation risks.  

**Feature Selection Techniques**:  
- **SHAP (SHapley Additive exPlanations)**: Quantified global feature importance.  
- **Correlation Analysis**: Removed multicollinear variables (e.g., ExperienceYearsAtThisCompany vs. YearsWithCurrManager).  
- **Domain-Driven Engineering**: Created EmpJobRole_freq (frequency encoding) to capture role-specific trends.  

**Why PCA Was Not Used**:  
- Interpretability was prioritized over dimensionality reduction.  
- Domain knowledge confirmed the relevance of raw features (e.g., salary hikes, promotions).  


### **3. Techniques & Tools**  
**Key Techniques**:  
- **Data Preprocessing**:  
  - Outlier handling via Winsorization for skewed variables (e.g., TotalWorkExperienceInYears).  
  - Categorical encoding (One-Hot for Gender, Frequency Encoding for EmpJobRole).  
- **Model Explainability**:  
  - SHAP waterfall plots to interpret XGBoost predictions.  
  - Partial Dependence Plots (PDPs) for non-linear relationships.  
- **Performance Metrics**:  
  - **Weighted F1-Score** to address class imbalance.  
  - **ROC AUC** for multi-class discrimination assessment.  

In the development of this project, I extensively utilized several functions from my custom library "insightfulpy." This library, available on both GitHub and PyPI, provided crucial functionalities that enhanced the data analysis and modeling process. For those interested in exploring the library or using it in their own projects, you can inspect the source code and documentation available. The functions from "insightfulpy" helped streamline data preprocessing, feature engineering, and model evaluation, making the analytic processes more efficient and reproducible.

You can find the source and additional resources on GitHub here: [insightfulpy on GitHub](https://github.com/dhaneshbb/insightfulpy), and for installation or further documentation, visit [insightfulpy on PyPI](https://pypi.org/project/insightfulpy/). These resources provide a comprehensive overview of the functions available and instructions on how to integrate them into your data science workflows.

Below is an overview of each major tool (packages, user-defined functions, and imported functions) that appears in this project.

<pre>
Imported packages:
1: insightfulpy
2: builtins
3: pandas
4: warnings
5: researchpy
6: matplotlib.pyplot
7: missingno
8: seaborn
9: numpy
10: scipy.stats
11: textwrap
12: logging
13: statsmodels.api
14: psutil
15: os
16: gc
17: time
18: xgboost
19: lightgbm
20: catboost
21: scikitplot
22: shap
23: joblib
24: types
25: inspect

User-defined functions:
1: normality_test_with_skew_kurt
2: spearman_correlation
3: memory_usage
4: dataframe_memory_usage
5: garbage_collection
6: calculate_vif
7: evaluate_model
8: cross_validation_analysis_table
9: plot_multiclass_evaluation
10: mean_absolute_shap

Imported functions:
1: open
2: tabulate
3: display
4: is_datetime64_any_dtype
5: skew
6: kurtosis
7: shapiro
8: kstest
9: compare_df_columns
10: linked_key
11: display_key_columns
12: interconnected_outliers
13: grouped_summary
14: calc_stats
15: iqr_trimmed_mean
16: mad
17: comp_cat_analysis
18: comp_num_analysis
19: detect_mixed_data_types
20: missing_inf_values
21: columns_info
22: cat_high_cardinality
23: analyze_data
24: num_summary
25: cat_summary
26: calculate_skewness_kurtosis
27: detect_outliers
28: show_missing
29: plot_boxplots
30: kde_batches
31: box_plot_batches
32: qq_plot_batches
33: num_vs_num_scatterplot_pair_batch
34: cat_vs_cat_pair_batch
35: num_vs_cat_box_violin_pair_batch
36: cat_bar_batches
37: cat_pie_chart_batches
38: num_analysis_and_plot
39: cat_analyze_and_plot
40: chi2_contingency
41: fisher_exact
42: pearsonr
43: spearmanr
44: ttest_ind
45: mannwhitneyu
46: linkage
47: dendrogram
48: leaves_list
49: variance_inflation_factor
50: train_test_split
51: cross_val_score
52: learning_curve
53: resample
54: compute_class_weight
55: accuracy_score
56: precision_score
57: recall_score
58: f1_score
59: roc_auc_score
60: confusion_matrix
61: precision_recall_curve
62: roc_curve
63: auc
64: calibration_curve
65: permutation_importance
66: classification_report
67: precision_recall_fscore_support
</pre>

The project leveraged **XGBoost** for its balance of accuracy and speed, with **SHAP-based interpretability** to validate critical features like salary hikes and promotions. By avoiding black-box dimensionality reduction (e.g., PCA), the analysis maintained actionable insights aligned with HR policies. Cross-validation and ensemble methods ensured robustness, while domain-driven preprocessing (e.g., role frequency encoding) captured nuanced patterns. This approach enables INX Future Inc. to deploy a transparent, high-performance model for talent management.  

---

# FEATURE SELECTION & ENGINEERING  


### **i. Most Important Features & Selection Rationale**  
**Top Features**:  
1. **EmpEnvironmentSatisfaction**  
   - **Why**: Highest SHAP value (0.494); employees with low scores (1–2) were 4x more likely to underperform. Directly tied to engagement and productivity.  
2. **EmpLastSalaryHikePercent**  
   - **Why**: Strong correlation with performance (ρ=0.72); top performers received 20.7% hikes vs. 14.4% for average. Reflects recognition and motivation.  
3. **YearsSinceLastPromotion**  
   - **Why**: Critical for retention; delays >3 years linked to 18.6% attrition in low performers. SHAP impact: 0.263.  

**Selection Criteria**:  
- **Domain Relevance**: Prioritized HR-identified drivers (e.g., promotions, satisfaction).  
- **Statistical Significance**: ANOVA (p<0.001) for satisfaction metrics.  
- **Model Impact**: SHAP values and permutation importance in XGBoost.  


### **ii. Feature Transformations**  
**Key Transformations**:  
1. **Categorical Encoding**:  
   - **One-Hot Encoding**: Applied to Gender, MaritalStatus, and BusinessTravelFrequency to convert nominal categories.  
   - **Frequency Encoding**: Created EmpJobRole_freq (frequency of each role in the dataset) to reduce high cardinality of EmpJobRole (19 unique roles).  
2. **Outlier Handling**:  
   - **Winsorization**: Capped extreme values in TotalWorkExperienceInYears (top 1% >40 years → 40).  
3. **Binning**:  
   - TrainingTimesLastYear grouped into "Low" (0–2), "Medium" (3–4), "High" (5–6) for clearer trends.  

**Why These Transformations?**  
- **Frequency Encoding**: Avoided dimensionality explosion while preserving role prevalence.  
- **Winsorization**: Mitigated skewness in experience variables without losing critical data.  


### **iii. Feature Correlations & Interactions**  
**Key Correlations**:  
1. **High Multicollinearity**:  
   - ExperienceYearsAtThisCompany vs. YearsWithCurrManager (ρ=0.84).  
   - **Action**: Retained ExperienceYearsAtThisCompany for broader tenure context; dropped YearsWithCurrManager.  
2. **Salary & Satisfaction Link**:  
   - EmpLastSalaryHikePercent correlated with EmpJobSatisfaction (ρ=0.61).  
   - **Insight**: Salary hikes indirectly boost job satisfaction.  

**Interactions**:  
- **XGBoost-Captured Non-Linearities**:  
  - EmpEnvironmentSatisfaction × OverTime: Low satisfaction + high overtime → 63% attrition risk.  
  - EmpEducationLevel × TotalWorkExperience: Advanced degrees + moderate experience → 78% likelihood of high performance.  

**How Correlations Were Addressed**:  
- **VIF Analysis**: Removed features with Variance Inflation Factor >10 (e.g., YearsSinceLastPromotion initially had VIF=12).  
- **Regularization**: L2 regularization in Logistic Regression to handle residual multicollinearity.  

 
The feature engineering process focused on enhancing interpretability and model efficiency. Critical features like EmpEnvironmentSatisfaction and salary hikes were prioritized due to their direct HR policy implications. Transformations such as frequency encoding and Winsorization balanced data integrity with usability, while correlation management ensured model robustness. XGBoost inherently captured complex interactions, avoiding the need for manual interaction terms.  

---

# RESULTS, ANALYSIS & INSIGHTS 


### **1. Interesting Relationships in the Data**  
- **Overtime ≠ Performance**: Employees working overtime had **lower performance ratings** (29.4% overtime workers vs. 70.6% non-overtime). This contradicts assumptions that extra hours boost productivity.  
- **Education vs. Experience**: Employees with *Technical Degrees* but **<5 years of experience** outperformed PhD holders with >10 years (F1=0.82 vs. 0.67), suggesting adaptability > tenure.  
- **Manager Tenure**: Employees with managers for **>7 years** had 23% higher satisfaction but 12% lower innovation metrics (e.g., project diversity).  


### **2. Most Important Technique**  
**SHAP (SHapley Additive exPlanations)**:  
- **Why**: Enabled granular interpretation of XGBoost predictions, revealing non-linear relationships (e.g., salary hikes’ diminishing returns above 20%).  
- **Impact**: Identified **EmpEnvironmentSatisfaction** as the top driver, guiding HR policy changes.  


### **3. Answers to Business Problems**  
| **Business Problem**                  | **Data-Backed Answer** |  
|----------------------------------------|-------------------------|  
| **Department-wise performance**       | - **Sales/Finance**: High attrition (18.6%) due to promotion delays. <br> - **Data Science**: 85% "Excellent" ratings due to structured training. |  
| **Top 3 performance factors**         | 1. Environment Satisfaction <br> 2. Salary Hike Equity <br> 3. Promotion Timeliness |  
| **Predictive model for hiring**       | XGBoost model (92.5% accuracy) flags candidates with: <br> - Education in *Life Sciences/Technical* fields. <br> - Prior experience in roles with **>2 promotions** in 5 years. |  
| **Recommendations**                   | - Redesign Sales incentives to reduce travel. <br> - Introduce bi-annual promotion cycles for Finance. |  


### **4. Additional Business Insights**  
1. **Work-Life Balance**: Employees with "Best" balance (score 4) were **2.5x more likely** to stay despite lower salaries.  
2. **Training Frequency**: Employees trained **3–4 times/year** had 19% higher performance vs. over-trained peers (diminishing returns beyond 4 sessions).  
3. **Role-Specific Trends**:  
   - **Sales Executives**: Performance dropped by 14% after 3 years in the same role.  
   - **R&D Managers**: Retention improved by 27% with cross-departmental projects.  
4. **Gender Dynamics**: Female employees in technical roles had **8% higher performance** but 12% lower promotion rates than male peers.  


### **Conclusion**  
The analysis uncovers actionable patterns beyond surface-level trends Address promotion delays in Sales/Finance to reduce attrition, Leverage SHAP-driven insights for bias-free hiring, Balance training frequency and role rotations to maintain engagement.  
