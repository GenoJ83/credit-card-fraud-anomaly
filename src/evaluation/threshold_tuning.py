"""Precision-Recall curve optimal threshold selection."""
import numpy as np

def find_optimal_threshold(scores, y_true, target_precision: float = 0.8):
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_thresh = thresholds[0]
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        prec = tp / (tp + fp + 1e-8)
        if prec >= target_precision:
            best_thresh = t
            break
    return float(round(best_thresh, 4))
