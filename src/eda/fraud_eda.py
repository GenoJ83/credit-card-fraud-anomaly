"""Imbalance metrics and PCA feature skewness profiling."""
from src.data.transaction_generator import generate_transactions

def profile_transaction_data():
    df = generate_transactions(4000)
    normal = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]
    stats = {
        "normal_mean_amount": round(float(normal["Amount"].mean()), 2),
        "fraud_mean_amount": round(float(fraud["Amount"].mean()), 2),
        "fraud_count": int(len(fraud)),
        "normal_count": int(len(normal))
    }
    print("Fraud vs Normal Profile:", stats)
    return stats

if __name__ == "__main__":
    profile_transaction_data()
