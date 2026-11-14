"""Cost-sensitive evaluation: AUPRC and Financial Cost-Benefit matrix."""
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

def compute_financial_cost(y_true, y_pred, amounts, fraud_investigation_cost: float = 5.0):
    false_positives = ((y_pred == 1) & (y_true == 0)).sum()
    false_negatives = ((y_pred == 0) & (y_true == 1))
    missed_fraud_amount = amounts[false_negatives].sum()

    investigation_overhead = false_positives * fraud_investigation_cost
    total_cost = missed_fraud_amount + investigation_overhead
    return {
        "missed_fraud_loss": float(round(missed_fraud_amount, 2)),
        "investigation_overhead": float(round(investigation_overhead, 2)),
        "total_financial_cost": float(round(total_cost, 2))
    }
