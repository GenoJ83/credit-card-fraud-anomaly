"""Dynamic rolling anomaly threshold adaptation engine."""
import numpy as np

class AdaptiveThresholdEngine:
    def __init__(self, base_threshold: float = 0.65, alpha: float = 0.1):
        self.current_threshold = base_threshold
        self.alpha = alpha

    def update_threshold(self, recent_scores: list, target_fp_rate: float = 0.01):
        if not recent_scores:
            return self.current_threshold
        target_quantile = 1.0 - target_fp_rate
        observed_quantile = float(np.quantile(recent_scores, target_quantile))
        self.current_threshold = (1 - self.alpha) * self.current_threshold + self.alpha * observed_quantile
        return round(self.current_threshold, 4)
