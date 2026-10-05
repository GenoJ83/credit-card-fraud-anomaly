# Credit Card Fraud Detection & Unsupervised Anomaly Engine

![Domain](https://img.shields.io/badge/Fintech-Fraud%20Prevention-blue)
![Architecture](https://img.shields.io/badge/Model-Isolation%20Forest%20%7C%20Autoencoder-purple)
![Imbalance](https://img.shields.io/badge/Class%20Imbalance-0.5%25-red)
![License](https://img.shields.io/badge/License-MIT-green)

Real-time transaction fraud screening engine designed for extreme class skew, combining statistical isolation trees with deep autoencoder reconstruction error.

## System Workflow
```mermaid
flowchart LR
    Tx[Card Transaction] --> Robust[RobustScaler]
    Robust --> Tree[Isolation Forest]
    Robust --> AE[Autoencoder Recon Error]
    Tree & AE --> Cost[Cost-Sensitive Decision Logic]
    Cost --> Alert{Action: APPROVE / BLOCK}
```

## Running the Microservice
```bash
uvicorn src.service.alert_service:app --reload
```
