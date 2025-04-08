import pandas as pd
from tabulate import tabulate
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)

# Load the Excel file
file_path = r"D:\INX_Future_Employee_Performance_Project\data\processed\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xlsx"
data = pd.read_excel(file_path)

from insightfulpy.eda import *
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, pearsonr, spearmanr, ttest_ind, mannwhitneyu, shapiro
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, learning_curve
)
from sklearn.preprocessing import (
     LabelEncoder, OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample, compute_class_weight
import time
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, 
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import scikitplot as skplt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

def normality_test_with_skew_kurt(df):
    # Initialize lists to store results for normal and non-normal distributions.
    normal_cols = []
    not_normal_cols = []
    # Loop over each numeric column in the dataframe.
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()  # Remove null values from the column.
        # Ensure there are at least 3 non-NA values to perform tests.
        if len(col_data) >= 3:
            # Select appropriate normality test based on sample size.
            if len(col_data) <= 5000:
                stat, p_value = shapiro(col_data)  # Shapiro-Wilk test for smaller samples.
                test_used = 'Shapiro-Wilk'
            else:
                # Kolmogorov-Smirnov test for larger samples, using sample mean and standard deviation.
                stat, p_value = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                test_used = 'Kolmogorov-Smirnov'
            # Calculate skewness and kurtosis of the column.
            col_skewness = skew(col_data)
            col_kurtosis = kurtosis(col_data)
            # Store test results in a dictionary.
            result = {
                'Column': col,
                'Test': test_used,
                'Statistic': stat,
                'p_value': p_value,
                'Skewness': col_skewness,
                'Kurtosis': col_kurtosis
            }
            # Classify the column based on the p-value.
            if p_value > 0.05:
                normal_cols.append(result)  # Considered normally distributed.
            else:
                not_normal_cols.append(result)  # Not normally distributed.
    # Create DataFrames for normal and not normal results and sort them.
    normal_df = (
        pd.DataFrame(normal_cols).sort_values(by='Column')
        if normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    not_normal_df = (
        pd.DataFrame(not_normal_cols).sort_values(by='p_value', ascending=False)
        if not_normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    # Display results for normal and not normal columns.
    print("\nNormal Columns (p > 0.05):")
    display(normal_df)
    print("\nNot Normal Columns (p ≤ 0.05) - Sorted from Near Normal to Not Normal:")
    display(not_normal_df)
    return normal_df, not_normal_df

def spearman_correlation(data, non_normal_cols, exclude_target=None, multicollinearity_threshold=0.8):
    # Exit if no non-normally distributed columns found.
    if non_normal_cols.empty:
        print("\nNo non-normally distributed numerical columns found. Exiting Spearman Correlation.")
        return
    # Prepare list of columns to analyze, excluding the target if specified.
    selected_columns = non_normal_cols['Column'].tolist()
    if exclude_target and exclude_target in selected_columns and pd.api.types.is_numeric_dtype(data[exclude_target]):
        selected_columns.remove(exclude_target)
    # Calculate Spearman correlation matrix.
    spearman_corr_matrix = data[selected_columns].corr(method='spearman')
    multicollinear_pairs = []
    # Identify pairs of variables with high correlation.
    for i, col1 in enumerate(selected_columns):
        for col2 in selected_columns[i+1:]:
            coef = spearman_corr_matrix.loc[col1, col2]
            if abs(coef) > multicollinearity_threshold:
                multicollinear_pairs.append((col1, col2, coef))
    # Display multicollinear pairs.
    print("\nVariables Exhibiting Multicollinearity (|Correlation| > {:.2f}):".format(multicollinearity_threshold))
    if multicollinear_pairs:
        for col1, col2, coef in multicollinear_pairs:
            print(f"- {col1} & {col2}: Correlation={coef:.4f}")
    else:
        print("No multicollinear pairs found.")
    # Configure and display heatmap for Spearman correlation matrix.
    annot_matrix = spearman_corr_matrix.round(2).astype(str)
    num_vars = len(selected_columns)
    fig_size = max(min(24, num_vars * 1.2), 10)  # Ensure reasonable figure size.
    annot_font_size = max(min(10, 200 / num_vars), 6)  # Adjust font size based on number of variables.
    plt.figure(figsize=(fig_size, fig_size * 0.75))
    sns.heatmap(
        spearman_corr_matrix,
        annot=annot_matrix,
        fmt='',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": annot_font_size},
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Spearman Correlation Matrix', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()

def calculate_vif(data, exclude_target='TARGET', multicollinearity_threshold=5.0):
    # Select only numeric columns, exclude target, and drop rows with missing values
    numeric_data = data.select_dtypes(include=[np.number]).drop(columns=[exclude_target], errors='ignore').dropna()
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_data.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_data.values, i) 
                       for i in range(numeric_data.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    high_vif = vif_data[vif_data['VIF'] > multicollinearity_threshold]
    low_vif = vif_data[vif_data['VIF'] <= multicollinearity_threshold]
    print(f"\nVariance Inflation Factor (VIF) Scores (multicollinearity_threshold = {multicollinearity_threshold}):")
    print("\nFeatures with VIF > threshold (High Multicollinearity):")
    if not high_vif.empty:
        print(high_vif.to_string(index=False))
    else:
        print("None. No features exceed the VIF threshold.")
    print("\nFeatures with VIF <= threshold (Low/No Multicollinearity):")
    if not low_vif.empty:
        print(low_vif.to_string(index=False))
    else:
        print("None. All features exceed the VIF threshold.")
    return vif_data, high_vif['Feature'].tolist()

print(data.shape)
for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

data.head().T

missing_inf_values(data)
print(f"\nNumber of duplicate rows: {data.duplicated().sum()}\n")
duplicates = data[data.duplicated()]
duplicates
show_missing(data)

data_negative_values = data.select_dtypes(include=[np.number]).lt(0).sum()
data_negative_values = data_negative_values[data_negative_values > 0].sort_values(ascending=False)
print("Columns with Negative Values (Sorted):\n", data_negative_values)

columns_info("Dataset Overview", data)

data = data.drop('EmpNumber', axis=1)
categorical_features = [
    'EmpEducationLevel', 'EmpEnvironmentSatisfaction', 'EmpJobInvolvement',
    'EmpJobLevel', 'EmpJobSatisfaction', 'EmpRelationshipSatisfaction',
    'EmpWorkLifeBalance', 'PerformanceRating', 'TrainingTimesLastYear'
]
for col in categorical_features:
    data[col] = data[col].astype('category')
# Convert object types to categorical where appropriate
object_features = [
    'Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment',
    'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition'
]
for col in object_features:
    data[col] = data[col].astype('category')

columns_info("Dataset Overview", data)

analyze_data(data)

data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
print(data_outlier_summary.shape)

plot_boxplots(data)
calculate_skewness_kurtosis(data)

data_outlier_summary

outlier_cols = ["YearsWithCurrManager", "ExperienceYearsInCurrentRole", "NumCompaniesWorked", "TotalWorkExperienceInYears","ExperienceYearsAtThisCompany", "YearsSinceLastPromotion"]
interconnected_outliers_df = interconnected_outliers(data, outlier_cols)

interconnected_outliers_df

# Perform domain-aware sanity checks
sanity_checks = {
    "Promotion vs Company Tenure": data["YearsSinceLastPromotion"] <= data["ExperienceYearsAtThisCompany"],
    "Company Tenure vs Total Experience": data["ExperienceYearsAtThisCompany"] <= data["TotalWorkExperienceInYears"],
    "Manager Tenure vs Company Tenure": data["YearsWithCurrManager"] <= data["ExperienceYearsAtThisCompany"],
    "Current Role Tenure vs Company Tenure": data["ExperienceYearsInCurrentRole"] <= data["ExperienceYearsAtThisCompany"]
}
# Create a DataFrame to track violations
sanity_check_df = pd.DataFrame({name: ~condition for name, condition in sanity_checks.items()})
sanity_check_df["Any Violation"] = sanity_check_df.any(axis=1)
# Count violations
violation_summary = sanity_check_df.sum()
# Extract violating records
violations = data[sanity_check_df["Any Violation"]]
# Output both summary and violating records
violation_summary, violations

# Capping TotalWorkExperienceInYears at 95th percentile
cap_value = data['TotalWorkExperienceInYears'].quantile(0.95)
data['TotalWorkExperienceInYears'] = data['TotalWorkExperienceInYears'].clip(upper=cap_value)

# Capping TotalWorkExperienceInYears at 95th percentile
cap_value = data['ExperienceYearsAtThisCompany'].quantile(0.95)
data['ExperienceYearsAtThisCompany'] = data['ExperienceYearsAtThisCompany'].clip(upper=cap_value)

# Define upper caps based on domain-informed thresholds for outlier treatment
caps = {
    "ExperienceYearsInCurrentRole": 14,
    "YearsSinceLastPromotion": 10,
    "YearsWithCurrManager": 14,
    "NumCompaniesWorked": 8
}
# Apply winsorization (capping at upper threshold)
for col, cap in caps.items():
    data[col] = data[col].clip(upper=cap)

data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
print(data_outlier_summary.shape)

data_outlier_summary

data_normal_df, data_not_normal_df = normality_test_with_skew_kurt(data)

spearman_correlation(data, data_not_normal_df, exclude_target='PerformanceRating', multicollinearity_threshold=0.8)

above_threshold, below_threshold = calculate_vif(data, exclude_target='PerformanceRating', multicollinearity_threshold=8.0)

# Drop multicollinear columns
data = data.drop(columns=["ExperienceYearsInCurrentRole", "YearsWithCurrManager"])
# Map education levels to estimated education years
education_years_map = {
    1: 11,  # Below College
    2: 13,  # College
    3: 16,  # Bachelor
    4: 18,  # Master
    5: 21   # Doctor
}
# Estimate years since first job
data["EducationYears"] = data["EmpEducationLevel"].cat.codes.map(lambda x: education_years_map.get(x + 1))
data["YearsSinceFirstJob"] = data["Age"] - data["EducationYears"]
# Compute total compensation: EmpHourlyRate * 2080 (working hours/year)
data["TotalCompensation"] = data["EmpHourlyRate"] * 2080
# Drop temporary helper column
data = data.drop(columns=["EducationYears"])
data = data.drop(columns=["Age", "EmpHourlyRate"])

data_normal_df1, data_not_normal_df1 = normality_test_with_skew_kurt(data)

spearman_correlation(data, data_not_normal_df1, exclude_target='PerformanceRating', multicollinearity_threshold=0.8)

above_threshold1, below_threshold1 = calculate_vif(data, exclude_target='PerformanceRating', multicollinearity_threshold=8.0)

# Identify predictors (excluding target)
categorical_predictors = [col for col in data.select_dtypes(include="category").columns if col != "PerformanceRating"]
numerical_predictors = [col for col in data.select_dtypes(include="number").columns]
# Chi-Square tests (Categorical vs. PerformanceRating)
chi_square_results = []
for col in categorical_predictors:
    try:
        contingency = pd.crosstab(data[col], data["PerformanceRating"])
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        chi_square_results.append((col, chi2, p))
    except Exception as e:
        chi_square_results.append((col, None, None))
# Kruskal-Wallis tests (Numerical vs. PerformanceRating)
kruskal_results = []
for col in numerical_predictors:
    try:
        groups = [group[col].values for name, group in data.groupby("PerformanceRating")]
        stat, p = stats.kruskal(*groups)
        kruskal_results.append((col, stat, p))
    except Exception as e:
        kruskal_results.append((col, None, None))
# Create DataFrames from results
chi_df = pd.DataFrame(chi_square_results, columns=["Feature", "Chi2 Stat", "p-value"])
kruskal_df = pd.DataFrame(kruskal_results, columns=["Feature", "Kruskal Stat", "p-value"])
# Format p-values to human-readable format with 4 decimal places or scientific notation
chi_df["p-value"] = chi_df["p-value"].apply(lambda x: f"{x:.4e}" if x is not None else "N/A")
kruskal_df["p-value"] = kruskal_df["p-value"].apply(lambda x: f"{x:.4e}" if x is not None else "N/A")

chi_df

kruskal_df

# Nominal (unordered) features
nominal_features = [
    "Gender", "EducationBackground", "MaritalStatus",
    "EmpDepartment", "EmpJobRole", "BusinessTravelFrequency"
]
# Separate high-cardinality from low-cardinality features
low_card_nominals = [col for col in nominal_features if data[col].nunique() <= 6]
high_card_nominals = [col for col in nominal_features if data[col].nunique() > 6]
# One-hot encode low-cardinality nominal features
data_encoded = pd.get_dummies(data, columns=low_card_nominals, drop_first=True)
# Frequency encode high-cardinality nominal features (e.g., EmpJobRole)
for col in high_card_nominals:
    freq_map = data[col].value_counts(normalize=True)
    data_encoded[col + "_freq"] = data[col].map(freq_map)
# Drop original high-cardinality nominal columns
data_encoded = data_encoded.drop(columns=high_card_nominals)
# Show result
data_encoded.head()

corrected_ordinal_mappings = {
    "EmpEducationLevel": [1, 2, 3, 4, 5],  
    "EmpEnvironmentSatisfaction": [1, 2, 3, 4],  
    "EmpJobInvolvement": [1, 2, 3, 4],  
    "EmpJobLevel": [1, 2, 3, 4, 5],  
    "EmpJobSatisfaction": [1, 2, 3, 4], 
    "EmpRelationshipSatisfaction": [1, 2, 3, 4],
    "EmpWorkLifeBalance": [1, 2, 3, 4],  
    "TrainingTimesLastYear": sorted(data["TrainingTimesLastYear"].dropna().unique()),  
    "PerformanceRating": [1, 2, 3, 4]  
}
# Apply ordinal encoding using defined category order
from pandas.api.types import CategoricalDtype
for col, order in corrected_ordinal_mappings.items():
    cat_type = CategoricalDtype(categories=order, ordered=True)
    data_encoded[col] = data[col].astype(cat_type).cat.codes
# Preview encoded values
data_encoded[list(corrected_ordinal_mappings.keys())].head()

# Binary features to encode: OverTime, Attrition
binary_map = {'Yes': 1, 'No': 0}
# Apply mapping
data_encoded["OverTime"] = data["OverTime"].map(binary_map)
data_encoded["Attrition"] = data["Attrition"].map(binary_map)
# Show result
data_encoded[["OverTime", "Attrition"]].head()

data_encoded["OverTime"] = data_encoded["OverTime"].astype("int8")
data_encoded["Attrition"] = data_encoded["Attrition"].astype("int8")

columns_info("Dataset Overview", data_encoded)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


X = data_encoded.drop(columns=["PerformanceRating"])
y = data_encoded["PerformanceRating"]

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
# Train Random Forest with class weighting for imbalance
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
# Permutation Importance (more reliable, slower)
perm_result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=5, random_state=42, n_jobs=-1
)
perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_result.importances_mean
}).sort_values('Importance', ascending=False)
# Combine ranks from both methods
feature_importance['Rank_RF'] = feature_importance['Importance'].rank(ascending=False)
perm_df['Rank_Perm'] = perm_df['Importance'].rank(ascending=False)
avg_rank = pd.merge(
    feature_importance[['Feature', 'Rank_RF']],
    perm_df[['Feature', 'Rank_Perm']],
    on='Feature'
)
avg_rank['Avg_Rank'] = avg_rank[['Rank_RF', 'Rank_Perm']].mean(axis=1)
# Select top N features
top_features = avg_rank.sort_values('Avg_Rank').head(15)['Feature'].tolist()
# Filter and sort for plotting
top_rf = feature_importance[feature_importance['Feature'].isin(top_features)].sort_values('Importance', ascending=True)
top_perm = perm_df[perm_df['Feature'].isin(top_features)].sort_values('Importance', ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=True)
axes[0].barh(top_rf['Feature'], top_rf['Importance'], color='skyblue')
axes[0].set_title('Random Forest Feature Importance', fontsize=14)
axes[0].set_xlabel('Importance Score')
axes[1].barh(top_perm['Feature'], top_perm['Importance'], color='salmon')
axes[1].set_title('Permutation Importance', fontsize=14)
axes[1].set_xlabel('Importance Score')
plt.tight_layout()
plt.show()
# Merge for comparison and correlation
combined = pd.merge(
    feature_importance[['Feature', 'Importance']],
    perm_df[['Feature', 'Importance']],
    on='Feature',
    suffixes=('_RF', '_Perm')
)
top_combined = combined.sort_values("Importance_RF", ascending=False).reset_index(drop=True)
correlation = np.corrcoef(combined['Importance_RF'], combined['Importance_Perm'])[0, 1]
print("\n Features by Both Methods (Sorted by RF Importance):")
print(top_combined.to_string(index=False))
print(f"\n Correlation between methods: {correlation:.2f}")

# Final feature list
final_features = [
    'EmpLastSalaryHikePercent',
    'EmpEnvironmentSatisfaction',
    'YearsSinceLastPromotion',
    'ExperienceYearsAtThisCompany',
    'EmpDepartment_Development',
    'EmpJobRole_freq',
    'EmpWorkLifeBalance',
    'OverTime',
    'EmpJobSatisfaction',
    'TotalWorkExperienceInYears',
    'DistanceFromHome',
    'PerformanceRating'  
]

# Filter the DataFrame
final_data = data_encoded[final_features]

missing_inf_values(final_data)
print(f"\nNumber of duplicate rows: {final_data.duplicated().sum()}\n")
duplicates = final_data[final_data.duplicated()]
duplicates
show_missing(final_data)

# Save to CSV
output_path = r"D:\INX_Future_Employee_Performance_Project\data\processed\prepared_data.csv"
final_data.to_csv(output_path, index=False)
print(f" Final data saved to: {output_path}")

