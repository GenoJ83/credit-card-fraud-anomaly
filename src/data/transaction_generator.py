"""Extreme class imbalance financial transactions generator (0.2% fraud)."""
import pandas as pd
import numpy as np

def generate_transactions(n_samples: int = 5000, fraud_ratio: float = 0.005) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # PCA feature simulations V1-V10
    normal_features = rng.normal(0, 1, size=(n_normal, 10))
    normal_amounts = np.round(rng.exponential(scale=50, size=n_normal), 2)
    normal_labels = np.zeros(n_normal, dtype=int)

    # Fraud features deviate significantly
    fraud_features = rng.normal(3.5, 2.0, size=(n_fraud, 10))
    fraud_amounts = np.round(rng.exponential(scale=250, size=n_fraud), 2)
    fraud_labels = np.ones(n_fraud, dtype=int)

    features = np.vstack([normal_features, fraud_features])
    amounts = np.concatenate([normal_amounts, fraud_amounts])
    labels = np.concatenate([normal_labels, fraud_labels])

    cols = [f"V{i+1}" for i in range(10)]
    df = pd.DataFrame(features, columns=cols)
    df["Amount"] = amounts
    df["Class"] = labels
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

if __name__ == "__main__":
    df = generate_transactions()
    print(f"Generated {len(df)} transactions. Frauds: {df['Class'].sum()} ({df['Class'].mean():.2%})")
