import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_model(model, X_train, y_train, X_test, y_test):
    import time

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
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
        # Multiclass: list of arrays (classes � samples � features)
        return np.mean([np.abs(class_shap).mean(0) for class_shap in shap_values], axis=0)
    elif len(shap_values.shape) == 3:
        # Multiclass: single array (samples � features � classes)
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
    import scikitplot as skplt

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
