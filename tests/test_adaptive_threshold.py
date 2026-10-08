"""Tests for adaptive fraud threshold."""
from src.evaluation.adaptive_threshold import AdaptiveThresholdEngine

def test_threshold_adaptation():
    engine = AdaptiveThresholdEngine(base_threshold=0.5)
    scores = [0.1, 0.2, 0.3, 0.85, 0.95]
    new_t = engine.update_threshold(scores)
    assert new_t > 0.5
