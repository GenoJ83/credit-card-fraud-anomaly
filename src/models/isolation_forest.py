"""Unsupervised Isolation Forest anomaly detector."""
from sklearn.ensemble import IsolationForest
import numpy as np
from src.data.transaction_generator import generate_transactions

class FraudIsolationForest:
    def __init__(self, contamination: float = 0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score_samples(self, X):
        # Invert score so higher = more anomalous
        return -self.model.score_samples(X)
