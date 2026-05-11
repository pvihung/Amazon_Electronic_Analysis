"""
Local dev:
    cd absa-api
    uvicorn app.main:app --reload --port 8080
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import ABSAService
from .schemas import BatchPredictRequest, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)

_service: ABSAService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = ABSAService()   # loads both models at startup
    yield
    _service = None


app = FastAPI(title="ABSA Inference API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return _service.predict(req.text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": type(exc).__name__, "message": str(exc)},
        )


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(req: BatchPredictRequest):
    if not req.texts:
        return []
    try:
        return _service.predict_batch(req.texts)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": type(exc).__name__, "message": str(exc)},
        )
