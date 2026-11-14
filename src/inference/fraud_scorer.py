"""High-throughput transaction scoring pipeline."""
import numpy as np
from src.models.isolation_forest import FraudIsolationForest

class TransactionScorer:
    def __init__(self, model: FraudIsolationForest = None, threshold: float = 0.65):
        self.model = model or FraudIsolationForest()
        self.threshold = threshold

    def score_transaction(self, features: np.ndarray):
        if features.ndim == 1:
            features = features.reshape(1, -1)
        score = float(self.model.score_samples(features)[0])
        is_fraud = score >= self.threshold
        return {
            "anomaly_score": round(score, 4),
            "flagged_fraud": bool(is_fraud),
            "action": "BLOCK_TRANSACTION" if is_fraud else "APPROVE"
        }
