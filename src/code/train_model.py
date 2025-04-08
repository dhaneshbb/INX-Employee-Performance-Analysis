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
from scipy.stats import (
    chi2_contingency, fisher_exact, pearsonr, spearmanr, 
    ttest_ind, mannwhitneyu, shapiro
)
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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