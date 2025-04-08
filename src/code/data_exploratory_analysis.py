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
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
#import dtale
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, pearsonr, spearmanr, ttest_ind, mannwhitneyu, shapiro
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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