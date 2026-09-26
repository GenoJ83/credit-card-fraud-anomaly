"""Z-score statistical thresholding heuristic baseline."""
import numpy as np
from src.data.transaction_generator import generate_transactions

def run_zscore_baseline():
    df = generate_transactions(2000)
    z_scores = np.abs((df["Amount"] - df["Amount"].mean()) / df["Amount"].std())
    preds = (z_scores > 3.0).astype(int)
    detected = ((preds == 1) & (df["Class"] == 1)).sum()
    print(f"Z-Score detected {detected} true frauds with {preds.sum()} flags.")
    return detected

if __name__ == "__main__":
    run_zscore_baseline()
