FROM python:3.11-slim

WORKDIR /app

# System deps needed by sentencepiece / protobuf builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than the default CUDA wheel)
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the DeBERTa base model so the container is self-contained
# and Cloud Run cold starts don't need outbound internet access.
RUN python -c "\
from transformers import AutoModel, AutoConfig; \
AutoConfig.from_pretrained('yangheng/deberta-v3-base-absa-v1.1'); \
AutoModel.from_pretrained('yangheng/deberta-v3-base-absa-v1.1'); \
print('DeBERTa base cached.')"

COPY app/ ./app/
COPY dapt_roberta/ ./models_artifacts/dapt_roberta/
COPY model1_multitask/ ./models_artifacts/model1_multitask/
COPY model2_sentiment/ ./models_artifacts/model2_sentiment/

ENV PORT=8080
EXPOSE 8080

# Single worker — models are ~1.5 GB each; multiple workers would OOM
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1"]
