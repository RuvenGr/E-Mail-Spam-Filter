# =============================================================================
# Spam Filter Implementation with Random Forest and SVM
#
# Description:
# This script trains and evaluates two machine learning models
# (Random Forest and Support Vector Machine) for classifying emails
# as "Spam" (+1) or "Not Spam" (-1).
#
# The primary goal is to achieve a False Positive Rate (FPR) of at most 0.2%
# while simultaneously maximizing the Spam Detection Rate (TPR).
#
# Methodology:
# 1. Splitting the data into a development set and a separate test set.
# 2. Applying 5-fold cross-validation on the development set to find a
#    robust and reliable classification threshold.
# 3. Selecting the final threshold based on a conservative strategy.
# 4. Performing a final training of the models on the entire development set.
# 5. Conducting an independent evaluation of the final models on the
#    previously untouched test set.
# =============================================================================

# ===== LIBRARIES =====
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# ===== 1. DATA LOADING FUNCTION =====
def load_data():
    """
    Loads the email data from the 'emails.mat' file.

    This function extracts the feature matrix 'X' and the target vector 'y',
    and transposes 'X' to fit the format expected by Scikit-learn
    (samples, features).

    Returns:
        tuple: A tuple (X, y) containing the features and labels.
    """
    data = scipy.io.loadmat("emails.mat")
    X = data['X'].T
    y = data['Y'].ravel()
    return X, y

# ===== 2. MODEL TRAINING FUNCTION =====
def train_models(X_train, y_train):
    """
    Initializes and trains a Random Forest and an SVM classifier.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.

    Returns:
        tuple: A tuple (rf_model, svm_model) containing the trained models.
    """
    # Random Forest: An ensemble model that is often robust and powerful.
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Support Vector Machine (SVM): A powerful classifier that finds an
    # optimal hyperplane between classes. probability=True is necessary
    # to obtain probability scores.
    svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return rf, svm

# ===== 3. THRESHOLD OPTIMIZATION FUNCTION =====
def find_best_effort_threshold(model, X_val, y_val, fpr_limit=0.002):
    """
    Finds the optimal classification threshold using a prioritized strategy.

    The function searches a range of possible thresholds and selects the
    best one based on two priorities:
    1. Find all thresholds that meet the FPR condition and, from that group,
       select the one with the highest True Positive Rate (TPR).
    2. If no threshold meets the condition, find the threshold that
       produces the absolute lowest FPR and maximize the TPR for that FPR.

    Args:
        model: The trained classification model (must support `predict_proba`).
        X_val (np.ndarray): Validation data features.
        y_val (np.ndarray): Validation data labels.
        fpr_limit (float): The maximum allowable False Positive Rate.

    Returns:
        float: The threshold determined to be optimal.
    """
    # Get the probabilities for the positive class (Spam)
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.5, 0.9999, 2000)

    results = []
    # Iterate over all potential thresholds
    for t in thresholds:
        # Classify based on the current threshold t
        y_pred = np.where(y_proba >= t, 1, -1)
        # Ignore thresholds that result in only one class prediction
        if len(np.unique(y_pred)) < 2: continue

        # Calculate the confusion matrix and its derived metrics
        cm = confusion_matrix(y_val, y_pred, labels=[1, -1])
        tp, fn, fp, tn = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append({'threshold': t, 'fpr': fpr, 'tpr': tpr})

    # Priority 1: Filter all results that meet the FPR condition
    valid_results = [res for res in results if res['fpr'] <= fpr_limit]

    if valid_results:
        # If there are valid candidates, choose the one with the highest TPR
        best_res = max(valid_results, key=lambda x: x['tpr'])
        print(f"      INFO: Optimal threshold ({best_res['threshold']:.4f}) found that meets the FPR condition (FPR: {best_res['fpr']:.3%}).")
        return best_res['threshold']
    else:
        # Priority 2: If no candidates meet the condition
        # Find the lowest possible FPR achievable with this model
        min_fpr = min(res['fpr'] for res in results)
        # Find all thresholds that produce this minimum FPR
        closest_results = [res for res in results if res['fpr'] == min_fpr]
        # From that group, choose the one with the highest TPR
        best_res = max(closest_results, key=lambda x: x['tpr'])

        print(f"      WARNING: FPR condition ({fpr_limit:.3%}) could not be met.")
        print(f"      -> Next-best threshold: {best_res['threshold']:.4f} (minimum FPR: {best_res['fpr']:.3%}, TPR: {best_res['tpr']:.2%})")
        return best_res['threshold']

# ===== 4. METRICS CALCULATION FUNCTION =====
def calculate_metrics(y_true, y_pred):
    """
    Calculates and collects the most important classification metrics.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The labels predicted by the model.

    Returns:
        dict: A dictionary containing the confusion matrix and metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    tp, fn, fp, tn = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average='binary', zero_division=0
    )
    # The spam detection rate is synonymous with Recall (True Positive Rate).
    spam_detection_rate = recall

    return {
        'confusion': cm, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'spam_detection_rate': spam_detection_rate
    }

# ===== 5. VISUALIZATION FUNCTION =====
def plot_final_confusion_matrix(metrics, model_name, threshold):
    """
    Creates and displays a detailed plot of the confusion matrix.

    Args:
        metrics (dict): The dictionary returned by `calculate_metrics`.
        model_name (str): The name of the model for the plot title.
        threshold (float): The threshold used, for the plot title.
    """
    labels = [
        f"TP\n{metrics['tp']}\n(Spam correctly detected)",
        f"FN\n{metrics['fn']}\n(Spam not detected)",
        f"FP\n{metrics['fp']}\n(Incorrectly marked as spam)",
        f"TN\n{metrics['tn']}\n(Correctly detected as 'Not Spam')"
    ]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        metrics['confusion'], annot=labels, fmt='',
        cmap='Blues' if 'SVM' in model_name else 'Greens',
        xticklabels=['Spam (+1)', 'Not Spam (-1)'],
        yticklabels=['Spam (+1)', 'Not Spam (-1)'],
        annot_kws={"size": 12}
    )
    plt.title(f"Final Confusion Matrix for {model_name}\n(Threshold: {threshold:.4f} on hold-out test set)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.show()

# ===== 6. MAIN PROGRAM =====
if __name__ == "__main__":
    # --- STEP 1: LOAD AND SPLIT DATA ---
    print("1. Loading data and creating development and test sets...")
    X, y = load_data()

    # Split into 80% development data and 20% test data.
    # The test set (hold-out) remains untouched until the final evaluation.
    # stratify=y ensures that the ratio of spam to non-spam is the same
    # in both sets as it is in the original dataset.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # --- STEP 2: K-FOLD CROSS-VALIDATION FOR THRESHOLD FINDING ---
    print("\n2. Starting 5-fold cross-validation to find the optimal threshold (FPR <= 0.2%)...")
    # KFold splits the development set into 5 different train/validation combinations.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rf_thresholds, svm_thresholds = [], []

    # This loop trains and validates the model 5 times on different splits of the data.
    for i, (train_index, val_index) in enumerate(kf.split(X_dev)):
        print(f"   Fold {i+1}/5:")
        X_train_fold, X_val_fold = X_dev[train_index], X_dev[val_index]
        y_train_fold, y_val_fold = y_dev[train_index], y_dev[val_index]

        rf_model, svm_model = train_models(X_train_fold, y_train_fold)

        print("    - Random Forest:")
        rf_thresh = find_best_effort_threshold(rf_model, X_val_fold, y_val_fold, fpr_limit=0.002)
        rf_thresholds.append(rf_thresh)

        print("    - SVM:")
        svm_thresh = find_best_effort_threshold(svm_model, X_val_fold, y_val_fold, fpr_limit=0.002)
        svm_thresholds.append(svm_thresh)

    # Determine the final threshold from the cross-validation results.
    # The MAXIMUM value from the 5 folds is chosen. This conservative strategy
    # prioritizes the avoidance of False Positives, as a higher threshold
    # requires the model to be more certain before classifying an email as spam.
    final_rf_threshold = np.max(rf_thresholds)
    final_svm_threshold = np.max(svm_thresholds)

    print(f"\nConservative (maximum) RF threshold: {final_rf_threshold:.4f}")
    print(f"Conservative (maximum) SVM threshold: {final_svm_threshold:.4f}")

    # --- STEP 3: FINAL TRAINING ---
    print("\n3. Training final models on the entire development set...")
    # The models are now trained on all 80% of the development data to
    # leverage the maximum amount of information before the final test.
    final_rf_model, final_svm_model = train_models(X_dev, y_dev)

    # --- STEP 4: FINAL EVALUATION ON THE TEST SET ---
    print("\n4. Evaluating final models on the untouched test set...")
    # The trained models are now applied to the 20% of data they have
    # never seen before. This provides a realistic estimate of performance
    # on new, unknown data.
    rf_proba_test = final_rf_model.predict_proba(X_test)[:, 1]
    rf_preds_test = np.where(rf_proba_test >= final_rf_threshold, 1, -1)
    rf_final_metrics = calculate_metrics(y_test, rf_preds_test)

    svm_proba_test = final_svm_model.predict_proba(X_test)[:, 1]
    svm_preds_test = np.where(svm_proba_test >= final_svm_threshold, 1, -1)
    svm_final_metrics = calculate_metrics(y_test, svm_preds_test)

    # --- STEP 5: PRINT AND VISUALIZE FINAL RESULTS ---
    print("\n" + "="*70)
    print("FINAL RESULTS ON THE HOLD-OUT TEST SET")
    print("="*70)

    print("\nRandom Forest Performance:")
    print(f"  Condition (FPR <= 0.2%): {'MET' if rf_final_metrics['fpr'] <= 0.002 else 'NOT MET'}")
    print(f"  Resulting FPR: {rf_final_metrics['fpr']:.3%} ({rf_final_metrics['fp']} out of {rf_final_metrics['fp'] + rf_final_metrics['tn']} legitimate emails)")
    print(f"  Spam detection rate (Recall/TPR): {rf_final_metrics['spam_detection_rate']:.2%}")
    print(f"  -> It is expected that {rf_final_metrics['spam_detection_rate']:.2%} of all future spam emails will be detected.")

    print("\nSVM Performance:")
    print(f"  Condition (FPR <= 0.2%): {'MET' if svm_final_metrics['fpr'] <= 0.002 else 'NOT MET'}")
    print(f"  Resulting FPR: {svm_final_metrics['fpr']:.3%} ({svm_final_metrics['fp']} out of {svm_final_metrics['fp'] + svm_final_metrics['tn']} legitimate emails)")
    print(f"  Spam detection rate (Recall/TPR): {svm_final_metrics['spam_detection_rate']:.2%}")
    print(f"  -> It is expected that {svm_final_metrics['spam_detection_rate']:.2%} of all future spam emails will be detected.")
    print("="*70)

    plot_final_confusion_matrix(rf_final_metrics, "Random Forest", final_rf_threshold)
    plot_final_confusion_matrix(svm_final_metrics, "SVM", final_svm_threshold)