"""Robust scaling for high-kurtosis transaction amounts."""
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

def build_fraud_preprocessor():
    # RobustScaler is less sensitive to extreme outliers
    return ColumnTransformer([
        ("amount_scaler", RobustScaler(), ["Amount"])
    ], remainder="passthrough")
