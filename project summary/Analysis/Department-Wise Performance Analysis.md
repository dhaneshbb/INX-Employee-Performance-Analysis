**Department-Wise Performance Analysis**  
**Project Code: 10281**  
**Document Code: CDS_Project_2_INX_Future_Emp_Data_V1.6**  
**Prepared for: INX Future Inc.**  

---

### **Executive Summary**  
This report analyzes department-wise employee performance at INX Future Inc., leveraging a dataset of 1,200 employees across 28 attributes. The goal is to identify departmental strengths, performance gaps, and actionable insights to address declining service delivery and client satisfaction. Key findings reveal significant disparities in performance ratings, satisfaction levels, and retention across departments, with actionable recommendations provided to enhance organizational outcomes.

---

### **Methodology**  
1. **Data Source**: Employee performance data from INX Future Inc., including demographic, job-related, satisfaction, and performance metrics.  
2. **Tools**: Python (Pandas, Seaborn, SciPy), statistical tests (Shapiro-Wilk, Spearman), and visualization techniques (heatmaps, bar charts).  
3. **Focus Areas**:  
   - Performance distribution by department.  
   - Correlation between satisfaction, tenure, and performance.  
   - Attrition trends and salary hike impacts.  

---

### **Key Findings: Department-Wise Performance**  

#### **1. Performance Rating Distribution**  
| **Department**              | **% Rating 2 (Good)** | **% Rating 3 (Excellent)** | **% Rating 4 (Outstanding)** | **Total Employees** |  
|-----------------------------|-----------------------|----------------------------|------------------------------|---------------------|  
| Development                 | 3.6%                  | 84.2%                      | 12.2%                        | 361                 |  
| Data Science                | 5.0%                  | 85.0%                      | 10.0%                        | 20                  |  
| Research & Development (R&D)| 19.8%                 | 68.2%                      | 11.9%                        | 343                 |  
| Sales                       | 23.3%                 | 67.3%                      | 9.4%                         | 373                 |  
| Finance                     | 30.6%                 | 61.2%                      | 8.2%                         | 49                  |  
| Human Resources (HR)        | 18.5%                 | 70.4%                      | 11.1%                        | 54                  |  

**Insights**:  
- **Development** and **Data Science** lead in high performance, with 84–85% of employees rated "Excellent" or "Outstanding."  
- **Sales** and **Finance** show higher proportions of "Good" ratings (23–31%), indicating performance variability.  
- **R&D** and **HR** exhibit moderate performance, with room for improvement in "Outstanding" ratings.  

---

#### **2. Critical Drivers of Departmental Performance**  
##### **a. Employee Satisfaction**  
- **Environment Satisfaction**:  
  - High-performing departments (Development, Data Science) report 75–80% employees in satisfaction levels 3–4 ("High" to "Very High").  
  - Sales and Finance show 40–50% of employees in lower satisfaction levels (1–2).  

- **Job Satisfaction**:  
  - Development: 65% employees rate satisfaction as 3–4.  
  - Sales: 45% employees rate satisfaction as 1–2, correlating with lower performance.  

##### **b. Salary Hikes & Recognition**  
- Employees rated "Outstanding" (Level 4) received **20.7% average salary hikes**, compared to 14–15% for lower ratings.  
- Development and Data Science employees with Level 4 ratings received hikes >20%, aligning with retention and motivation.  

##### **c. Attrition Trends**  
| **Department**  | **Attrition Rate** | **Primary Drivers**                          |  
|------------------|--------------------|----------------------------------------------|  
| Sales            | 18.6%              | Low satisfaction, longer promotion gaps.    |  
| Finance          | 15.3%              | Limited training, moderate salary hikes.    |  
| Development      | 5.1%               | High satisfaction, frequent promotions.     |  

---

#### **3. Experience & Tenure Dynamics**  
- **Development**: Shorter tenure (avg. 6.7 years) but higher performance, indicating efficient talent utilization.  
- **Sales**: Longer tenure (avg. 9.1 years) but stagnant promotions (avg. 3.7 years since last promotion).  

---

### **Recommendations**  
1. **Enhance Satisfaction in Underperforming Departments**:  
   - **Sales/Finance**: Implement mentorship programs and flexible work policies to improve environment/job satisfaction.  
   - Conduct quarterly feedback sessions to address grievances.  

2. **Align Rewards with Performance**:  
   - Introduce tiered salary hikes for "Outstanding" ratings in Sales/Finance to mirror Development’s structure.  

3. **Reduce Promotion Gaps**:  
   - Set clear promotion timelines in Sales/R&D to prevent stagnation (e.g., max 2–3 years between promotions).  

4. **Targeted Training**:  
   - Prioritize technical upskilling in Sales (e.g., CRM tools) and leadership training in Finance.  

5. **Retention Strategies for High Performers**:  
   - Offer career path customization in Development/Data Science to retain top talent.  

---

### **Conclusion**  
Departmental performance at INX Future Inc. is strongly influenced by satisfaction, recognition, and career progression. By addressing gaps in underperforming departments and replicating best practices from high-performing teams, INX can restore service delivery standards and client satisfaction while maintaining its reputation as a top employer.  

**Next Steps**: Proceed to model training to predict employee performance and validate these insights. 