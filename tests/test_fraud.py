import pytest
pytest.importorskip("torch")
"""Unit tests for transaction generator, Autoencoder, and cost metrics."""
import torch
from src.data.transaction_generator import generate_transactions
from src.models.autoencoder import FraudAutoencoder
from src.evaluation.cost_metrics import compute_financial_cost
import numpy as np

def test_data_generation():
    df = generate_transactions(500, fraud_ratio=0.02)
    assert len(df) == 500
    assert df["Class"].sum() == 10

def test_autoencoder_reconstruction():
    ae = FraudAutoencoder(input_dim=11, latent_dim=4)
    x = torch.randn(8, 11)
    recon = ae(x)
    assert recon.shape == (8, 11)

def test_financial_cost():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0, 1, 1])
    amounts = np.array([100.0, 50.0, 200.0])
    cost = compute_financial_cost(y_true, y_pred, amounts, fraud_investigation_cost=5.0)
    assert cost["missed_fraud_loss"] == 100.0
    assert cost["investigation_overhead"] == 5.0
