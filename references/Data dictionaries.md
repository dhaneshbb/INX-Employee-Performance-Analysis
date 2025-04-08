### Data Dictionary for INX Future Inc. Employee Performance Dataset  

The dataset contains 1,200 employee records with 28 features (27 after dropping EmpNumber). Below is the additional structured breakdown of variables, including descriptions, data types, encoded values (for categorical features), and domain categorization.  

#### 1. Personal Demographics  
| Variable              | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|---------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| Age                     | Age of the employee                                                            | int64       | Range: 18–60                                                                              |  
| Gender                  | Gender of the employee                                                         | category    | Male, Female                                                                          |  
| EducationBackground     | Educational background of the employee                                         | category    | Life Sciences, Medical, Marketing, Technical Degree, Other, Human Resources   |  
| MaritalStatus           | Marital status of the employee                                                 | category    | Married, Single, Divorced                                                           |  
| DistanceFromHome        | Distance (in km) from home to workplace                                        | int64       | Range: 1–29                                                                               |  
| EmpEducationLevel       | Education level (ordinal)                                                      | category    | Encoded: 1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor     |  



#### 2. Job-Related Information  
| Variable              | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|---------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| EmpDepartment           | Department where the employee works                                            | category    | Sales, Development, Research & Development, Human Resources, Finance, Data Science |  
| EmpJobRole              | Job role of the employee                                                       | category    | 19 roles, e.g., Sales Executive, Developer, Manager R&D, Research Scientist        |  
| BusinessTravelFrequency | Frequency of business travel                                                   | category    | Travel_Rarely, Travel_Frequently, Non-Travel                                         |  
| EmpJobLevel             | Job level within the organization (ordinal)                                    | category    | 1 (Entry) to 5 (Executive)                                                            |  
| OverTime                | Whether the employee works overtime                                            | category    | Yes, No                                                                               |  



#### 3. Satisfaction & Engagement  
| Variable                      | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|-----------------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| EmpEnvironmentSatisfaction      | Satisfaction with the work environment (ordinal)                               | category    | Encoded: 1: Low, 2: Medium, 3: High, 4: Very High                              |  
| EmpJobInvolvement               | Level of job involvement (ordinal)                                             | category    | Encoded: 1: Low, 2: Medium, 3: High, 4: Very High                              |  
| EmpJobSatisfaction              | Satisfaction with the job role (ordinal)                                       | category    | Encoded: 1: Low, 2: Medium, 3: High, 4: Very High                              |  
| EmpRelationshipSatisfaction     | Satisfaction with workplace relationships (ordinal)                            | category    | Encoded: 1: Low, 2: Medium, 3: High, 4: Very High                              |  
| EmpWorkLifeBalance              | Balance between work and personal life (ordinal)                               | category    | Encoded: 1: Bad, 2: Good, 3: Better, 4: Best                                   |  



#### 4. Work Experience  
| Variable                      | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|-----------------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| NumCompaniesWorked              | Number of companies the employee has worked for                                 | int64       | Range: 0–9                                                                                |  
| TotalWorkExperienceInYears      | Total work experience in years                                                  | int64       | Range: 0–40                                                                               |  
| ExperienceYearsAtThisCompany    | Years of experience at INX Future Inc.                                          | int64       | Range: 0–40                                                                               |  
| ExperienceYearsInCurrentRole    | Years in the current role                                                       | int64       | Range: 0–18                                                                               |  
| YearsSinceLastPromotion         | Years since the last promotion                                                  | int64       | Range: 0–15                                                                               |  



#### 5. Compensation & Benefits  
| Variable                  | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|-------------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| EmpHourlyRate               | Hourly wage of the employee                                                     | int64       | Range: 30–100                                                                             |  
| EmpLastSalaryHikePercent    | Percentage of the last salary hike                                              | int64       | Range: 11–25                                                                              |  



#### 6. Professional Development  
| Variable              | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|--------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| TrainingTimesLastYear  | Number of training sessions attended last year                                  | category    | Encoded: 0, 1, 2, 3, 4, 5, 6                                            |  



#### 7. Organizational Structure  
| Variable              | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|--------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| YearsWithCurrManager   | Years with the current manager                                                  | int64       | Range: 0–17                                                                               |  



#### 8. Performance Metrics  
| Variable              | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|--------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| Attrition              | Whether the employee left the company                                           | category    | Yes, No                                                                               |  
| PerformanceRating      | Target Variable: Performance rating (ordinal)                               | category    | Encoded: 2: Good, 3: Excellent, 4: Outstanding                                  |  



#### 9. Identification  
| Variable              | Description                                                                 | Data Type | Possible Values/Notes                                                                 |  
|--------------------------|---------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------|  
| EmpNumber              | Unique employee identifier (dropped in analysis)                                | object      | High cardinality (1,200 unique values)                                                    |  



### Key Notes:  
- Target Variable: PerformanceRating is the primary outcome for predictive modeling.  
- Encoded Variables: Ordinal features (e.g., EmpEducationLevel, satisfaction scores) use numerical labels with explicit human-readable meanings.  
- High Cardinality: EmpJobRole (19 unique roles) and EmpNumber (unique identifier) were excluded from modeling to avoid noise.  
- Multicollinearity: Experience-related variables (e.g., ExperienceYearsAtThisCompany, YearsWithCurrManager) are highly correlated.  
