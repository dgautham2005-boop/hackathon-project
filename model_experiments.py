"""
model_experiments.py
Author: Person 2 (Model Training & Validation)
Description: Path-safe group-aware cross-validation, metric evaluation, 
and pure JSON artifact serialization.
"""

import os
import gzip
import json
import numpy as np
import sklearn as sk
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(filename):
    path = os.path.join(BASE_DIR, filename)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def load_json(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_metrics(y_true_prob, y_pred_prob, y_true_val, y_pred_val, cap=2000.0):
    brier = brier_score_loss(y_true_prob, np.clip(y_pred_prob, 0.0, 1.0))
    y_true_capped = np.clip(y_true_val, 0.0, cap)
    y_pred_capped = np.clip(y_pred_val, 0.0, cap)
    capped_mse = mean_squared_error(y_true_capped, y_pred_capped)
    combined_score = brier + (capped_mse / (cap ** 2))
    return {
        "brier_score": float(brier),
        "capped_mse": float(capped_mse),
        "combined_score": float(combined_score)
    }

def main():
    print("=== Starting Model Training & Validation Pipeline ===")
    
    train_data = load_data("training.json.gz")
    practice_data = load_data("practice.json.gz")
    ds_info = load_json("dataset_info.json")
    
    groups = np.array(ds_info["train_groups"])
    
    X_train, y_prob_train, y_val_train = features.extract_features(train_data)
    X_practice, y_prob_practice, y_val_practice = features.extract_features(practice_data)
    
    gkf = GroupKFold(n_splits=5)
    best_score = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_val_train, groups)):
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        p_tr, p_va = y_prob_train[train_idx], y_prob_train[val_idx]
        v_tr, v_va = y_val_train[train_idx], y_val_train[val_idx]
        
        prob_model = Ridge(alpha=1.0).fit(X_tr, p_tr)
        val_model = Ridge(alpha=1.0).fit(X_tr, v_tr)
        
        preds_prob = np.clip(prob_model.predict(X_va), 0.0, 1.0)
        preds_val = np.clip(val_model.predict(X_va), 0.0, 2000.0)
        
        metrics = compute_metrics(p_va, preds_prob, v_va, preds_val)
        
        if metrics["combined_score"] < best_score:
            best_score = metrics["combined_score"]

    print(f"Best Group-Aware Cross-Validation Combined Score: {best_score:.5f}")
    
    # Fit final models on full training data
    final_prob_model = Ridge(alpha=1.0).fit(X_train, y_prob_train)
    final_val_model = Ridge(alpha=1.0).fit(X_train, y_val_train)
    
    # Save parameters in pure JSON (Strictly NO pickle/joblib)
    output_artifact = {
        "model_type": "two_output_ridge_optimized",
        "parameters": {
            "prob_coef": final_prob_model.coef_.tolist(),
            "prob_intercept": float(final_prob_model.intercept_),
            "val_coef": final_val_model.coef_.tolist(),
            "val_intercept": float(final_val_model.intercept_)
        },
        "validation_metrics": {
            "best_cv_combined_score": float(best_score)
        }
    }
    
    out_path = os.path.join(BASE_DIR, "improved_model.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_artifact, f, indent=2)
        
    print("Successfully generated and saved improved_model.json")

if __name__ == "__main__":
    main()
    