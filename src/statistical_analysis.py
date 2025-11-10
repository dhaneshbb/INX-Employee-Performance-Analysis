import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kstest, kurtosis, shapiro, skew
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
                stat, p_value = shapiro(
                    col_data
                )  # Shapiro-Wilk test for smaller samples.
                test_used = "Shapiro-Wilk"
            else:
                # Kolmogorov-Smirnov test for larger samples, using sample mean and standard deviation.
                stat, p_value = kstest(
                    col_data, "norm", args=(col_data.mean(), col_data.std())
                )
                test_used = "Kolmogorov-Smirnov"
            # Calculate skewness and kurtosis of the column.
            col_skewness = skew(col_data)
            col_kurtosis = kurtosis(col_data)
            # Store test results in a dictionary.
            result = {
                "Column": col,
                "Test": test_used,
                "Statistic": stat,
                "p_value": p_value,
                "Skewness": col_skewness,
                "Kurtosis": col_kurtosis,
            }
            # Classify the column based on the p-value.
            if p_value > 0.05:
                normal_cols.append(result)  # Considered normally distributed.
            else:
                not_normal_cols.append(result)  # Not normally distributed.
    # Create DataFrames for normal and not normal results and sort them.
    normal_df = (
        pd.DataFrame(normal_cols).sort_values(by="Column")
        if normal_cols
        else pd.DataFrame(
            columns=["Column", "Test", "Statistic", "p_value", "Skewness", "Kurtosis"]
        )
    )
    not_normal_df = (
        pd.DataFrame(not_normal_cols).sort_values(by="p_value", ascending=False)
        if not_normal_cols
        else pd.DataFrame(
            columns=["Column", "Test", "Statistic", "p_value", "Skewness", "Kurtosis"]
        )
    )
    # Display results for normal and not normal columns.
    print("\nNormal Columns (p > 0.05):")
    display(normal_df)
    print("\nNot Normal Columns (p d 0.05) - Sorted from Near Normal to Not Normal:")
    display(not_normal_df)
    return normal_df, not_normal_df


def spearman_correlation(
    data, non_normal_cols, exclude_target=None, multicollinearity_threshold=0.8
):
    # Exit if no non-normally distributed columns found.
    if non_normal_cols.empty:
        print(
            "\nNo non-normally distributed numerical columns found. Exiting Spearman Correlation."
        )
        return
    # Prepare list of columns to analyze, excluding the target if specified.
    selected_columns = non_normal_cols["Column"].tolist()
    if (
        exclude_target
        and exclude_target in selected_columns
        and pd.api.types.is_numeric_dtype(data[exclude_target])
    ):
        selected_columns.remove(exclude_target)
    # Calculate Spearman correlation matrix.
    spearman_corr_matrix = data[selected_columns].corr(method="spearman")
    multicollinear_pairs = []
    # Identify pairs of variables with high correlation.
    for i, col1 in enumerate(selected_columns):
        for col2 in selected_columns[i + 1 :]:
            coef = spearman_corr_matrix.loc[col1, col2]
            if abs(coef) > multicollinearity_threshold:
                multicollinear_pairs.append((col1, col2, coef))
    # Display multicollinear pairs.
    print(
        "\nVariables Exhibiting Multicollinearity (|Correlation| > {:.2f}):".format(
            multicollinearity_threshold
        )
    )
    if multicollinear_pairs:
        for col1, col2, coef in multicollinear_pairs:
            print(f"- {col1} & {col2}: Correlation={coef:.4f}")
    else:
        print("No multicollinear pairs found.")
    # Configure and display heatmap for Spearman correlation matrix.
    annot_matrix = spearman_corr_matrix.round(2).astype(str)
    num_vars = len(selected_columns)
    fig_size = max(min(24, num_vars * 1.2), 10)  # Ensure reasonable figure size.
    annot_font_size = max(
        min(10, 200 / num_vars), 6
    )  # Adjust font size based on number of variables.
    plt.figure(figsize=(fig_size, fig_size * 0.75))
    sns.heatmap(
        spearman_corr_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": annot_font_size},
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Spearman Correlation Matrix", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()


def calculate_vif(data, exclude_target="TARGET", multicollinearity_threshold=5.0):
    # Select only numeric columns, exclude target, and drop rows with missing values
    numeric_data = (
        data.select_dtypes(include=[np.number])
        .drop(columns=[exclude_target], errors="ignore")
        .dropna()
    )
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_data.columns
    vif_data["VIF"] = [
        variance_inflation_factor(numeric_data.values, i)
        for i in range(numeric_data.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    high_vif = vif_data[vif_data["VIF"] > multicollinearity_threshold]
    low_vif = vif_data[vif_data["VIF"] <= multicollinearity_threshold]
    print(
        f"\nVariance Inflation Factor (VIF) Scores (multicollinearity_threshold = {multicollinearity_threshold}):"
    )
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
    return vif_data, high_vif["Feature"].tolist()
