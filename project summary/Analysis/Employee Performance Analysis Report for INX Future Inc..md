**Employee Performance Analysis Report for INX Future Inc.**  
**Project Code: 10281 | Document Code: CDS_Project_2_INX_Future_Emp_Data_V1.6**  

---

### **Executive Summary**  
INX Future Inc., a leader in data analytics and automation, faces declining employee performance metrics, impacting service delivery and client satisfaction. This report synthesizes data-driven insights from 1,200 employees to identify root causes and proposes actionable solutions. Key findings reveal departmental disparities, critical performance drivers, and a validated predictive model for hiring. Recommendations focus on enhancing satisfaction, aligning rewards, and leveraging analytics to sustain INX’s reputation as a top employer without penalizing employees.  

---

### **1. Department-Wise Performances**  
#### **Performance Ratings**  
| **Department**          | **% "Excellent/Outstanding"** | **% "Good"** | **Attrition Rate** |  
|-------------------------|-------------------------------|---------------|--------------------|  
| Development & Data Science | 96.4%                        | 3.6–5.0%      | 5.1%               |  
| R&D                      | 80.1%                        | 19.8%         | 14.2%              |  
| Sales                    | 76.7%                        | 23.3%         | 18.6%              |  
| Finance                  | 69.4%                        | 30.6%         | 15.3%              |  

**Key Insights**:  
- **High Performers**: Development and Data Science excel due to frequent promotions, high satisfaction (75–80% in "High/Very High" scores), and competitive salary hikes (>20%).  
- **Underperformers**: Sales and Finance struggle with low satisfaction (40–50% in "Low" scores), longer promotion gaps (3.7 years avg. in Sales), and unrealistic targets.  
- **Attrition Drivers**: Low satisfaction and stagnation in Sales/Finance; high satisfaction in Development reduces turnover.  

---

### **2. Top 3 Factors Affecting Employee Performance**  
Identified via **XGBoost/Random Forest models (92.5% accuracy)**:  

#### **1. Employee Environment Satisfaction**  
- **Impact**: Employees with low satisfaction (score 1–2) are **3x more likely to underperform**.  
- **Evidence**: SHAP value of **0.4945** (highest weight in models).  

#### **2. Last Salary Hike Percentage**  
- **Impact**: Employees with hikes <15% show **40% lower performance** vs. those with hikes >20%.  
- **Evidence**: SHAP value of **0.372**; strong correlation with retention.  

#### **3. Years Since Last Promotion**  
- **Impact**: Employees unpromoted for 3+ years exhibit **35% lower performance**.  
- **Evidence**: SHAP value of **0.2631**; frequent promotions in Development drive 25% higher performance.  

---

### **3. Predictive Model for Employee Performance**  
**Model**: XGBoost (92.5% accuracy, ROC AUC 0.976).  
**Key Inputs**:  
- `EmpEnvironmentSatisfaction`, `EmpLastSalaryHikePercent`, `YearsSinceLastPromotion`.  
- Secondary factors: `EmpWorkLifeBalance`, `TotalWorkExperienceInYears`.  

**Deployment**:  
- **Usage**: Predicts performance ratings (Low/Average/High) for new hires.  
- **Action**: Prioritize candidates with:  
  - High environment satisfaction in prior roles.  
  - Salary hikes aligned with performance (≥15%).  
  - Balanced tenure (3–5 years per role).  

---

### **4. Recommendations to Improve Performance**  
#### **A. Address Environmental Satisfaction**  
- **Action**: Quarterly surveys, ergonomic workspace upgrades, and mentorship programs in Sales/Finance.  
- **Goal**: Increase "High/Very High" satisfaction from 60% to 80% in 12 months.  

#### **B. Revise Compensation & Promotions**  
- **Action**:  
  - Link salary hikes to performance (e.g., ≥15% for "Excellent" ratings).  
  - Implement **biennial promotions** in Sales/R&D to reduce stagnation.  
- **Goal**: Cut promotion delays by 50% and align rewards with output.  

#### **C. Department-Specific Interventions**  
- **Sales/Finance**:  
  - Reduce burnout via revised targets (cut quarterly goals by 15%).  
  - Train on advanced tools (CRM, financial modeling).  
- **R&D/Data Science**: Offer innovation grants for high-impact projects.  

#### **D. Predictive Hiring & Retention**  
- **Action**: Use the XGBoost model to screen hires and flag attrition risks.  
- **Goal**: Improve hiring accuracy by 20% and retention by 10%.  

#### **E. Leadership Development**  
- **Action**: Train managers in performance coaching and conflict resolution.  
- **Goal**: Boost manager-related satisfaction by 25%.  

---

### **Conclusion**  
By addressing environmental satisfaction, aligning rewards with performance, and deploying data-driven hiring strategies, INX Future Inc. can reverse declining trends while preserving employee morale. The predictive model ensures sustainable growth by identifying high-potential candidates and systemic risks. These steps align with INX’s ethos as a top employer and position it for long-term success.  

**Next Steps**:  
1. Implement recommendations within 6 months.  
2. Monitor progress via quarterly performance dashboards.  
3. Expand data collection to refine model accuracy.  

