from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str


class BatchPredictRequest(BaseModel):
    texts: list[str]


class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str      # "Positive" or "Negative"
    confidence: float


class PredictResponse(BaseModel):
    text: str
    is_related: bool
    is_technical: bool
    sentiment_aspects: list[AspectSentiment]  # aspects with confident sentiment
    mentioned_aspects: list[str]              # detected but below M2 confidence threshold
