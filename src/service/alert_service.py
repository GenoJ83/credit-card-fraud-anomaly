"""FastAPI transaction fraud screening microservice."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
from src.inference.fraud_scorer import TransactionScorer

app = FastAPI(title="Card Fraud Alert Service", version="1.0.0")

class TransactionPayload(BaseModel):
    Amount: float = Field(..., ge=0.01)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float

scorer = TransactionScorer()

@app.get("/health")
def health():
    return {"status": "active", "service": "fraud-detection"}

@app.post("/screen")
def screen_transaction(payload: TransactionPayload):
    feats = np.array([payload.V1, payload.V2, payload.V3, payload.V4, payload.V5, payload.V6, payload.V7, payload.V8, payload.V9, payload.V10, payload.Amount])
    return scorer.score_transaction(feats)
