import pandas as pd
from tabulate import tabulate
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)

# Path to the Excel file
file_path = r'D:\INX_Future_Employee_Performance_Project\data\raw\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls'
# available sheet
xl = pd.ExcelFile(file_path)
print("Available sheets:", xl.sheet_names)

# Path for the new Excel workbook copy
copy_file_path = r'D:\INX_Future_Employee_Performance_Project\data\processed\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xlsx'
writer = pd.ExcelWriter(copy_file_path, engine='xlsxwriter')
# Iterate through each sheet and copy it
for sheet_name in xl.sheet_names:
    # Load the sheet into a DataFrame
    df = xl.parse(sheet_name)
    # Write the DataFrame to a new file
    df.to_excel(writer, sheet_name=sheet_name, index=False)
# Excel writer and output the Excel file.
writer.close()

# Load the 'Data Definitions' sheet from the copied file
data_definitions = pd.read_excel(copy_file_path, sheet_name='Data Definitions')

# Correctly fill NaN values forward in the first column to associate each definition with its category
data_definitions['Unnamed: 0'] = data_definitions['Unnamed: 0'].ffill()

# Clean up the DataFrame by removing rows where the second column is NaN
data_definitions.dropna(subset=['Unnamed: 1'], inplace=True)

# cleaned content 
print("Cleaned Content of 'Data Definitions' Sheet in Markdown Table:")
print(data_definitions.to_markdown(index=False, tablefmt='pipe'))

# Load the 'INX_Future_Inc_Employee_Perform' sheet into a DataFrame
data = pd.read_excel(copy_file_path, sheet_name='INX_Future_Inc_Employee_Perform')

print(data.shape)
for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

from insightfulpy.eda import *

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from scipy.stats import (
    chi2_contingency, fisher_exact, pearsonr, spearmanr, 
    ttest_ind, mannwhitneyu, shapiro
)
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from insightfulpy.eda import *

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    learning_curve, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample, compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc, 
    classification_report, precision_recall_fscore_support
)
from sklearn.calibration import calibration_curve
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
import shap
from tabulate import tabulate
import time
import os

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

def evaluate_model(model, X_train, y_train, X_test, y_test):
    import time
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time  

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except:
            roc_auc = None
    else:
        roc_auc = None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1_metric = f1_score(y_test, y_pred, average='weighted')
    acc_train = accuracy_score(y_train, y_pred_train)
    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=strat_kfold, scoring='f1_weighted').mean()
    overfit = acc_train - acc
    return {
        "Training Time (sec)": round(train_time, 3),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1_metric,
        "ROC AUC": roc_auc,
        "CV F1-Score": cv_f1,
        "Train Accuracy": acc_train,
        "Overfit (Train - Test Acc)": overfit
    }

def mean_absolute_shap(shap_values):
    """Handle both binary and multiclass SHAP values"""
    if isinstance(shap_values, list):
        # Multiclass: list of arrays (classes × samples × features)
        return np.mean([np.abs(class_shap).mean(0) for class_shap in shap_values], axis=0)
    elif len(shap_values.shape) == 3:
        # Multiclass: single array (samples × features × classes)
        return np.abs(shap_values).mean(axis=(0, 2))  # Mean across samples and classes
    else:
        # Binary classification
        return np.abs(shap_values).mean(axis=0)

def cross_validation_analysis_table(model, X_train, y_train, cv_folds=5, scoring_metric="f1_weighted"):
    strat_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=strat_kfold, scoring=scoring_metric)   
    cv_results_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(cv_folds)],
        "F1-Score": scores
    })
    # Append Mean and Standard Deviation
    cv_results_df.loc[len(cv_results_df)] = ["Mean", np.mean(scores)]
    cv_results_df.loc[len(cv_results_df)] = ["Standard Deviation", np.std(scores)]
    
    return cv_results_df
    
def plot_multiclass_evaluation(model, X_test, y_test, class_labels=None):
    y_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    if class_labels is None:
        class_labels = [f"Class {i}" for i in np.unique(y_test)]
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    try:
        roc_auc = roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro")
    except:
        roc_auc = None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ROC Curve
    if roc_auc:
        skplt.metrics.plot_roc(y_test, y_probs, ax=axes[0, 0])
        axes[0, 0].set_title(f"ROC Curve (Macro AUC = {roc_auc:.3f})")
    else:
        axes[0, 0].text(0.5, 0.5, "ROC Curve not available", ha='center', va='center', fontsize=12)
        axes[0, 0].axis("off")
    # Precision-Recall Curve
    skplt.metrics.plot_precision_recall_curve(y_test, y_probs, ax=axes[0, 1])
    axes[0, 1].set_title("Precision-Recall Curve")
    # Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(ax=axes[1, 0], cmap="Blues")
    axes[1, 0].set_title("Confusion Matrix")
    # Normalized Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_labels).plot(ax=axes[1, 1], cmap="Blues")
    axes[1, 1].set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()

import psutil
import os
import gc

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

def dataframe_memory_usage(df):
    mem_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"DataFrame Memory Usage: {mem_usage:.2f} MB")
    return mem_usage

def garbage_collection():
    gc.collect()
    memory_usage()

if __name__ == "__main__":
    memory_usage()

dataframe_memory_usage(data)

print(data.shape)
for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

data.head().T

detect_mixed_data_types(data)

cat_high_cardinality(data)

missing_inf_values(data)
print(f"\nNumber of duplicate rows: {data.duplicated().sum()}\n")
duplicates = data[data.duplicated()]
duplicates
show_missing(data)

data_negative_values = data.select_dtypes(include=[np.number]).lt(0).sum()
data_negative_values = data_negative_values[data_negative_values > 0].sort_values(ascending=False)
print("Columns with Negative Values (Sorted):\n", data_negative_values)

data.dtypes.value_counts()

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

cat_analyze_and_plot(data, 'PerformanceRating')

cat_analyze_and_plot(data, 'EmpDepartment', "PerformanceRating")

grouped_summary(data, groupby='PerformanceRating')

columns_info("Dataset Overview", data)

num_summary(data)

cat_summary(data)

kde_batches(data, batch_num=1)
box_plot_batches(data, batch_num=1)
qq_plot_batches(data, batch_num=1)

cat_bar_batches(data, batch_num=1)
cat_bar_batches(data, batch_num=2)

cat_pie_chart_batches(data, batch_num=1)
cat_pie_chart_batches(data, batch_num=2)

sns.set_theme(style="whitegrid")
int_cols = data.select_dtypes(include='int64').drop(columns=['EmpNumber'], errors='ignore')
sns.pairplot(int_cols, kind='reg', diag_kind='kde', plot_kws={'line_kws': {'color': 'red'}})
plt.show()

cat_vs_cat_pair_batch(data, pair_num=15, batch_num=1)
cat_vs_cat_pair_batch(data, pair_num=15, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=0, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=0, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=1, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=1, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=2, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=2, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=3, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=3, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=4, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=4, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=5, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=5, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=6, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=6, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=7, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=7, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=8, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=8, batch_num=2)

num_vs_cat_box_violin_pair_batch(data, pair_num=9, batch_num=1)
num_vs_cat_box_violin_pair_batch(data, pair_num=9, batch_num=2)

data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
print(data_outlier_summary.shape)

data_outlier_summary

data_normal_df, data_not_normal_df = normality_test_with_skew_kurt(data)

spearman_correlation(data, data_not_normal_df, exclude_target='PerformanceRating', multicollinearity_threshold=0.8)

# Load the Excel file
file_path = r"D:\INX_Future_Employee_Performance_Project\data\processed\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xlsx"
data = pd.read_excel(file_path)

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

# Path to the processed data
file_path = r"D:\INX_Future_Employee_Performance_Project\data\processed\prepared_data.csv"
# Load the data
data = pd.read_csv(file_path)

columns_info("Dataset Overview", data)

print(data.shape)
for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

data.head().T

# Split the full dataset (assuming `final_data` includes the features + target column)
X = data.drop(columns=['PerformanceRating'])
y = data['PerformanceRating']

# Map target labels from [1, 2, 3] → [0, 1, 2]
label_mapping = {1: 0, 2: 1, 3: 2}
y_mapped = y.map(label_mapping)

# Train-Test Split (Stratified to preserve class distribution)
X_train, X_test, y_train_mapped, y_test_mapped = train_test_split(
    X, y_mapped, test_size=0.2, stratify=y_mapped, random_state=42
)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train_mapped.shape}, y_test: {y_test_mapped.shape}")
output_dir = r"D:\INX_Future_Employee_Performance_Project\data\processed"
os.makedirs(output_dir, exist_ok=True)
# Save splits
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train_mapped.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test_mapped.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

classes = np.unique(y_train_mapped)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_mapped)
mapped_class_weights = dict(zip(classes, weights))
catboost_weights = list(weights)  # for CatBoost

mapped_class_weights

catboost_weights

# Fit scaler on train set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define which models need scaled input
scaling_sensitive_models = [
    "Logistic Regression (Multinomial)",
    "Support Vector Machine (OVO)",
    "KNN (Weighted)",
    "MLP Classifier"
]

base_models = {
    "Logistic Regression (Multinomial)": LogisticRegression(multi_class='multinomial', class_weight=mapped_class_weights, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight=mapped_class_weights, random_state=42),
    "LightGBM": LGBMClassifier(class_weight=mapped_class_weights, objective='multiclass', num_class=3, random_state=42),
    "CatBoost": CatBoostClassifier(class_weights=catboost_weights, objective='MultiClass', classes_count=3, silent=True, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(class_weight=mapped_class_weights, random_state=42),
    "XGBoost": XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    "MLP Classifier": MLPClassifier(random_state=42),
    "Support Vector Machine (OVO)": SVC(decision_function_shape='ovo', probability=True, random_state=42),
    "KNN (Weighted)": KNeighborsClassifier(weights='distance'),
    "Naive Bayes (Adjusted)": GaussianNB(priors=[0.16, 0.72, 0.12])
}

all_model_results = {}
for name, model in base_models.items():
    print(f"\n Training & Evaluating: {name}")
    # Choose scaled or original inputs based on model type
    if name in scaling_sensitive_models:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    result = evaluate_model(model, X_tr, y_train_mapped, X_te, y_test_mapped)
    all_model_results[name] = result
results_df = pd.DataFrame.from_dict(all_model_results, orient='index')
base_results = results_df.sort_values(by="F1-Score", ascending=False)

base_results

param_grids = {
    "Random Forest": {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    "CatBoost": {
    'iterations': [200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [5, 6, 7],
    'l2_leaf_reg': [1, 3, 5],
    'bootstrap_type': ['Bernoulli', 'MVS'],  # Valid bootstrap types (without 'subsample')
    'class_weights': [catboost_weights]  # Use catboost weights
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [3.0],
        'eval_metric': ['mlogloss'],
        'objective': ['multi:softprob']
    },
    "LightGBM": {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 5, 10],
        'reg_alpha': [0, 0.1, 0.2],
        'reg_lambda': [0, 0.1, 0.2],
        'class_weight': [mapped_class_weights]
    },
    "MLP Classifier": {
        'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'batch_size': [32, 64, 128],
        'max_iter': [500, 1000]
    },
    "Logistic Regression (Multinomial)": {
        'multi_class': ['multinomial'],
        'solver': ['saga', 'lbfgs'],
        'class_weight': [mapped_class_weights],
        'max_iter': [1000, 2000],
        'penalty': ['l2']
    },
    "Support Vector Machine (OVO)": {
        'kernel': ['rbf', 'poly'],
        'C': [0.1, 1, 10],
        'decision_function_shape': ['ovo', 'ovr'],
        'probability': [True],
        'random_state': [42]
    },
    "KNN (Weighted)": {
        'n_neighbors': [5, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    "Naive Bayes (Adjusted)": {
        'priors': [[0.16, 0.72, 0.12]]
    },
    "HistGradientBoosting": {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 6, 7],
        'max_iter': [100, 200, 300],
        'min_samples_leaf': [20, 50],
        'early_stopping': ['auto'],
        'class_weight': [mapped_class_weights]
    }
}

# Define search parameters
n_iter_search = 50  # Number of iterations for RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Dictionary to store the best models after tuning
best_models = {}
# Loop through each base model and tune hyperparameters
for name, model in base_models.items():  # Use 'base_models' instead of 'base_estimators'
    print(f"\nTuning {name}...")
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[name],  # Hyperparameter grid for each model
        scoring='f1_weighted',  # Weighted F1-score for multiclass classification
        n_iter=n_iter_search,  # Number of iterations to search
        cv=cv,  # Cross-validation splitting strategy
        verbose=2,  # Show progress
        n_jobs=-1,  # Use all available CPU cores
        random_state=42,  # For reproducibility
        refit=True  # Refitting the best model with the best parameters
    )
    # Fit the model using RandomizedSearchCV
    random_search.fit(X_train, y_train_mapped)
    # Store the best model after hyperparameter tuning
    best_models[name] = random_search.best_estimator_
    # Output the best hyperparameters and best CV F1-score
    print(f"\nBest parameters for {name}:\n{random_search.best_params_}")
    print(f"Best CV F1-Score: {random_search.best_score_:.4f}\n")

all_model_results = {}
for name, model in best_models.items():
    print(f"\nEvaluating {name} on Test Data...")
    result = evaluate_model(model, X_train, y_train_mapped, X_test, y_test_mapped)
    all_model_results[name] = result
# Show results in a DataFrame
results_df = pd.DataFrame.from_dict(all_model_results, orient='index')
tune_results = results_df.sort_values(by="F1-Score", ascending=False)

tune_results

# Define the best models for ensemble
top_models = [
    ("XGBoost", best_models["XGBoost"]),
    ("Random Forest", best_models["Random Forest"]),
    ("HistGradientBoosting", best_models["HistGradientBoosting"]),
    ("CatBoost", best_models["CatBoost"]),
    ("LightGBM", best_models["LightGBM"]),
]
# Define the VotingClassifier with soft voting
voting_ensemble = VotingClassifier(estimators=top_models, voting='soft')

# Fit the voting ensemble on the training data
voting_ensemble.fit(X_train, y_train_mapped)
# Evaluate using the evaluate_model() function
ensemble_result = evaluate_model(voting_ensemble, X_train, y_train_mapped, X_test, y_test_mapped)
# Show results in a DataFrame
ensemble_df = pd.DataFrame([ensemble_result], index=["Voting Ensemble"])

ensemble_df

# Define the base models for stacking
base_learners = [
    ("XGBoost", best_models["XGBoost"]),
    ("Random Forest", best_models["Random Forest"]),
    ("HistGradientBoosting", best_models["HistGradientBoosting"]),
    ("CatBoost", best_models["CatBoost"]),
    ("LightGBM", best_models["LightGBM"]),
]
# Define the meta-model (Logistic Regression)
meta_model = LogisticRegression(class_weight=mapped_class_weights, max_iter=2000, random_state=42)
# Initialize the StackingClassifier
stacking_ensemble = StackingClassifier(estimators=base_learners, final_estimator=meta_model)
# Fit the stacking ensemble on the training data
stacking_ensemble.fit(X_train, y_train_mapped)
# Evaluate using the evaluate_model() function
stacking_result = evaluate_model(stacking_ensemble, X_train, y_train_mapped, X_test, y_test_mapped)
# Show results in a DataFrame
stacking_df = pd.DataFrame([stacking_result], index=["Stacking Ensemble"])

stacking_df

# Add a category for each set of results
base_results["Category"] = "Base Models"
tune_results["Category"] = "Tuned Models"
ensemble_df["Category"] = "Ensemble Models"
stacking_df["Category"] = "Stacking Models"
# Concatenate all the DataFrames together
all_results = pd.concat([base_results, tune_results, ensemble_df, stacking_df])
# Reorder columns to have 'Category' first
all_results = all_results[['Category'] + [col for col in all_results.columns if col != 'Category']]
# Display the final combined DataFrame
all_results = all_results.sort_values(by="F1-Score", ascending=False)  # Sort by F1-Score
all_results

# Extract feature importance from each model
xgb_importance = best_models["XGBoost"].feature_importances_
rf_importance = best_models["Random Forest"].feature_importances_
lgb_importance = best_models["LightGBM"].feature_importances_
catboost_importance = best_models["CatBoost"].get_feature_importance()
# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "XGBoost": xgb_importance,
    "Random Forest": rf_importance,
    "LightGBM": lgb_importance,
    "CatBoost": catboost_importance
})
# Sort by XGBoost importance
importance_df = importance_df.sort_values(by="XGBoost", ascending=False)
# Display the table in pipe-separated format using tabulate
print(tabulate(importance_df, headers='keys', tablefmt='pipe', showindex=False))

# Get SHAP values correctly formatted
explainer_xgb = shap.TreeExplainer(best_models["XGBoost"])
xgb_shap = explainer_xgb.shap_values(X_train)
explainer_rf = shap.TreeExplainer(best_models["Random Forest"])
rf_shap = explainer_rf.shap_values(X_train)
explainer_lgb = shap.TreeExplainer(best_models["LightGBM"])
lgb_shap = explainer_lgb.shap_values(X_train)
explainer_cat = shap.TreeExplainer(best_models["CatBoost"])
catboost_shap = explainer_cat.shap_values(X_train)
# Create importance DataFrame
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "XGBoost": mean_absolute_shap(xgb_shap),
    "Random Forest": mean_absolute_shap(rf_shap),
    "LightGBM": mean_absolute_shap(lgb_shap),
    "CatBoost": mean_absolute_shap(catboost_shap)
}).sort_values("XGBoost", ascending=False)
# Format and display
print(tabulate(importance_df.round(4), headers='keys', tablefmt='pipe', showindex=False))

classes = np.unique(y_train_mapped)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_mapped)
class_weights = dict(zip(classes, weights))

sample_weights = np.array([class_weights[label] for label in y_train_mapped])
final_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(classes),
    subsample=0.9,
    colsample_bytree=1.0,
    learning_rate=0.01,
    max_depth=3,
    n_estimators=300,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
final_model.fit(X_train, y_train_mapped, sample_weight=sample_weights)

final_result = evaluate_model(final_model, X_train, y_train_mapped, X_test, y_test_mapped)
final_df = pd.DataFrame([final_result], index=["Final XGBoost Model"])
final_df

# Usage
cv_results_table = cross_validation_analysis_table(final_model, X_train, y_train_mapped, cv_folds=5)
# View results
print(cv_results_table)

# Predict on test set
y_pred = final_model.predict(X_test)
# Classification Report
print("\n Classification Report:\n")
print(classification_report(y_test_mapped, y_pred, target_names=["Rating 1", "Rating 2", "Rating 3"]))
# Confusion Matrix
print(" Confusion Matrix (rows = Actual, columns = Predicted):\n")
cm = confusion_matrix(y_test_mapped, y_pred)
print(cm)

plot_multiclass_evaluation(final_model, X_test, y_test_mapped, class_labels=["Rating 1", "Rating 2", "Rating 3"])

import joblib
# Save the final trained model
joblib.dump(final_model, "XGBClassifier_model_multiclass.pkl")
print(" Model saved successfully!")

# Load the trained model
final_model = joblib.load("XGBClassifier_model_multiclass.pkl")
# Predict the class labels on the test set
y_pred_final = final_model.predict(X_test)
# Show classification report
print("\n Classification Report (Multiclass):")
print(classification_report(y_test_mapped, y_pred_final, target_names=["Rating 1", "Rating 2", "Rating 3"]))

import types
import inspect
user_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ == '__main__']
imported_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ != '__main__']
imported_pkgs = [name for name in globals() if isinstance(globals()[name], types.ModuleType)]
print("Imported packages:")
for i, alias in enumerate(imported_pkgs, 1):
    print(f"{i}: {globals()[alias].__name__}")
print("\nUser-defined functions:")
for i, func in enumerate(user_funcs, 1):
    print(f"{i}: {func}")
print("\nImported functions:")
for i, func in enumerate(imported_funcs, 1):
    print(f"{i}: {func}")