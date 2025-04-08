# Employee Performance Analysis for INX Future Inc.  
**Project Code: 10281**  
**Document Code: CDS_Project_2_INX_Future_Emp_Data_V1.6**  

---

## Project Overview  
INX Future Inc., a leading data analytics and automation solutions provider, faces declining employee performance impacting client satisfaction. This project analyzes employee data (1,200 records) to identify performance bottlenecks, predict employee success, and provide actionable recommendations. The goal is to address systemic issues without penalizing employees, preserving INX’s reputation as a top employer.  

**Key Objectives**:  
1. Analyze department-wise performance disparities.  
2. Identify top factors influencing employee performance.  
3. Build a predictive model for hiring decisions.  
4. Recommend strategies to improve performance and retention.  

---

## Key Findings  

### 1. **Department-Wise Performance**  
| **Department**          | **% "Excellent/Outstanding"** | **Attrition Rate** |  
|-------------------------|-------------------------------|--------------------|  
| Development & Data Science | 96.4%                        | 5.1%               |  
| Sales                    | 76.7%                        | 18.6%              |  
| Finance                  | 69.4%                        | 15.3%              |  

**Insights**:  
- **High Performers**: Development teams thrive due to frequent promotions (>20% salary hikes).  
- **Underperformers**: Sales/Finance face low satisfaction (40–50% "Low" scores) and stagnation (3.7-year promotion gaps).  

### 2. **Top 3 Performance Drivers**  
1. **Employee Environment Satisfaction** (SHAP: 0.4945)  
   - Low satisfaction employees are **3x more likely to underperform**.  
2. **Last Salary Hike Percentage** (SHAP: 0.372)  
   - Employees with <15% hikes show **40% lower performance**.  
3. **Years Since Last Promotion** (SHAP: 0.2631)  
   - Delays >3 years lead to **35% lower performance**.  

### 3. **Predictive Model**  
- **Algorithm**: XGBoost (92.5% accuracy, ROC AUC: 0.976).  
- **Key Inputs**:  
  - EmpEnvironmentSatisfaction, EmpLastSalaryHikePercent, YearsSinceLastPromotion.  
- **Usage**: Predicts performance ratings (Low/Average/High) for hiring.  

---

##  Directory Structure  
```  
Root
├── README.md
├── requirements.txt
├── data
│   ├── raw
│   │   └── INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xlsx
│   ├── processed
│   │   ├── X_test.csv
│   │   ├── X_train.csv
│   │   ├── prepared_data.csv
│   │   ├── y_test.csv
│   │   └── y_train.csv
├── project_summary
│   ├── Department-Wise Performance Analysis.md
│   ├── Top 3 Factors Affecting Employee Performance.md
│   ├── Employee Performance Prediction Model.md
│   ├── Recommendations to Improve Employee Performance at INX Future Inc..md
│   └── Analysis Report for INX Future Inc..md
│   └── Requirement
│       ├── CDS_Project_2_INX_Future_Emp_Data_V1.6.pdf
│       └── requirements.txt
├── references
│   ├── IABAC_CDS_Project_Submission_Guidelines_V1.2.pdf
│   ├── References.md
│   └── CDS_Project_2_INX_Future_Emp_Data_V1.6.pdf
├── src
│   ├── Data Processing
│   │   ├── data_exploratory_analysis.ipynb
│   │   ├── data_processing.ipynb
│   │   └── .ipynb_checkpoints
│   │       ├── 1. Exploratory Data Analysis (EDA)-checkpoint.ipynb
│   │       └── 2. Data_processing-checkpoint.ipynb
│   ├── models
│   │   ├── train_model.ipynb
│   │   ├── predict_model.ipynb
│   │   └── XGBClassifier_model_multiclass.pkl
│   │   └── .ipynb_checkpoints
│   │       └── 3. Model_Development-checkpoint.ipynb
│   ├── visualization
│   │   ├── plot_gallery.md
│   │   ├── plots
│   │   │   ├── image_0.png
│   │   │   ├── image_1.png
│   │   │   └── image_2.png
│   │   │   └── ... (more images)
│   ├── full_analysis
│   │   ├── Full_Analysis.ipynb
│   │   ├── XGBClassifier_model_multiclass.pkl
│   │   └── .ipynb_checkpoints
│   │       └── Full_Analysis-checkpoint.ipynb
└── external
  
```

---

## Usage  
1. **Data Preprocessing**:  
   - Clean and encode data using src/2. Data_processing.ipynb.  
   - Handle class imbalance via stratified sampling.  
2. **Model Training**:  
   - Train the XGBoost model with optimized hyperparameters (src/3. Model_Development.ipynb).  
3. **Predictions**:  
   - Use the trained model (XGBClassifier_model_multiclass.pkl) to predict new hires' performance.  


---

## Model Summary & Comparison 


### **Top Performing Models**  

| **Model**               | **Accuracy** | **F1-Score** | **Training Time (sec)** | **Overfit (Train - Test Acc)** | **ROC AUC** |  
|-------------------------|--------------|--------------|-------------------------|--------------------------------|-------------|  
| **XGBoost (Tuned)**     | 92.50%       | 92.42%       | 0.337                   | 0.0083                        | 0.9756      |  
| **Random Forest (Tuned)**| 92.50%       | 92.34%       | 0.827                   | 0.0531                        | 0.9731      |  
| **Voting Ensemble**      | 92.08%       | 91.99%       | 3.843                   | 0.0677                        | 0.9744      |  
| **Stacking Ensemble**    | 91.25%       | 91.19%       | 21.492                  | 0.0552                        | 0.9672      |  


### **Key Observations**  
1. **XGBoost (Tuned)**:  
   - **Best Overall Performance**: Achieved 92.5% accuracy and 92.42% F1-Score with minimal overfitting (0.0083).  
   - **Efficiency**: Fast training time (0.337 sec) and highest ROC AUC (0.9756).  
   - **Class-Specific Performance**:  
     - **Rating 3 (Outstanding)**: 100% precision but 77% recall, indicating challenges in identifying top performers.  
     - **Rating 2 (Average)**: Dominated predictions with 97% recall.  

2. **Random Forest (Tuned)**:  
   - **Balanced Performance**: Matched XGBoost's accuracy (92.5%) but with slightly higher overfitting (0.0531).  
   - **Feature Importance**: Prioritized EmpEnvironmentSatisfaction (27.1% impact) and EmpLastSalaryHikePercent (29.9%).  

3. **Ensemble Models**:  
   - **Voting Ensemble**: Combined predictions of top 5 models, achieving 92.08% accuracy with stable generalization.  
   - **Stacking Ensemble**: Used logistic regression as meta-model, showing robustness but slower training (21.492 sec).  

4. **Underperformers**:  
   - **KNN (Weighted)**: Lowest accuracy (75%) and high overfitting (25%).  
   - **Naive Bayes**: Struggled with imbalanced classes (82.92% accuracy, F1=82.57%).  


### **Critical Features Influencing Performance**  
Identified via **SHAP Analysis**:  
1. **EmpEnvironmentSatisfaction** (SHAP: 0.4945 in XGBoost).  
2. **EmpLastSalaryHikePercent** (SHAP: 0.372).  
3. **YearsSinceLastPromotion** (SHAP: 0.2631).  

**Insights**:  
- Employees with low environment satisfaction are **3x more likely to underperform**.  
- Salary hikes >20% correlate with 40% higher performance.  
- Promotion delays >3 years reduce performance by 35%.  

---

## Results & Recommendations  
### **Strategic Recommendations**  
- **Revise Compensation**: Link salary hikes to performance (≥15% for "Excellent" ratings).  
- **Reduce Promotion Gaps**: Implement biennial promotions in Sales/Finance.  
- **Enhance Work Environment**: Quarterly surveys and ergonomic upgrades in low-satisfaction departments.  
- **Predictive Hiring**: Use the XGBoost model to prioritize candidates with high satisfaction and balanced tenure.  
