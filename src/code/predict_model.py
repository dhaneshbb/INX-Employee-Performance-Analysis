import pandas as pd
from tabulate import tabulate
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)

# Path to the processed data
file_path = r"D:\INX_Future_Employee_Performance_Project\data\processed\prepared_data.csv"
# Load the data
data = pd.read_csv(file_path)

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
import statsmodels.api as sm
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    learning_curve, StratifiedKFold
)
from sklearn.utils import resample, compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc, 
    classification_report, precision_recall_fscore_support
)
from sklearn.calibration import calibration_curve
import scikitplot as skplt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import shap
from tabulate import tabulate
import time
import os

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

def cross_validation_analysis_table(model, X_train, y_train, cv_folds=5, scoring_metric="f1_weighted"):
    """
    This function performs cross-validation on the given model and generates a table with the evaluation scores.
    """
    # Ensure y_train is the correct format (integer-encoded or categorical)
    if not np.issubdtype(y_train.dtype, np.integer):
        print("Warning: Target variable `y_train` should be integer-encoded for multiclass classification.")
    strat_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    # Perform cross-validation and calculate scores
    scores = cross_val_score(model, X_train, y_train, cv=strat_kfold, scoring=scoring_metric)
    
    # If the result is NaN, there might be an issue with the data or scoring metric
    if np.any(np.isnan(scores)):
        print(f"Error: Cross-validation resulted in NaN values. Please check the data and try with another scoring metric.")
        return None
    # Create a DataFrame to display the results
    cv_results_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(cv_folds)],
        "F1-Score": scores
    })
    # Add Mean and Standard Deviation of scores
    cv_results_df.loc["Mean"] = ["Mean", np.mean(scores)]
    cv_results_df.loc["Std"] = ["Standard Deviation", np.std(scores)]
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

# Feature importance from tree splits
xgb_importance = final_model.feature_importances_
# SHAP importance (mean absolute value)
explainer_xgb = shap.TreeExplainer(final_model.get_booster())
xgb_shap = explainer_xgb.shap_values(X_train)
shap_importance = mean_absolute_shap(xgb_shap)
# Combine df
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Tree-Based": xgb_importance,
    "SHAP": shap_importance
}).sort_values("SHAP", ascending=False)
print(tabulate(importance_df.round(4), headers='keys', tablefmt='pipe', showindex=False))
# Subplot bar chart
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
# Plot
axs[0].barh(importance_df["Feature"], importance_df["Tree-Based"])
axs[0].set_title("XGBoost Tree-Based Importance")
axs[0].invert_yaxis()
axs[1].barh(importance_df["Feature"], importance_df["SHAP"])
axs[1].set_title("XGBoost SHAP Importance")
axs[1].invert_yaxis()
plt.tight_layout()
plt.show()

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