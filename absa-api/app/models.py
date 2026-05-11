import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, RobertaModel
from peft import LoraConfig, get_peft_model

from .schemas import AspectSentiment, PredictResponse

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent / "models_artifacts"



# Model architectures


class _M1(nn.Module):
    """RoBERTa + LoRA encoder with three Sequential(Dropout, Linear) heads.

    Architecture reverse-engineered from model_best.pt state dict keys:
      encoder.*         — peft-wrapped RobertaModel
      head1.1.*         — is_related    (Sequential index 1 = Linear)
      head2.1.*         — is_technical
      head3.1.*         — aspects multi-label (n_aspects outputs)
    """

    def __init__(
        self,
        backbone_path: Path,
        n_aspects: int,
        lora_r: int,
        lora_alpha: int,
        lora_target: list[str],
    ):
        super().__init__()
        roberta = RobertaModel.from_pretrained(str(backbone_path))
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target,
            lora_dropout=0.0,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.encoder = get_peft_model(roberta, lora_cfg)
        hidden = roberta.config.hidden_size
        self.head1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.head2 = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.head3 = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, n_aspects))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return (
            torch.sigmoid(self.head1(cls)).squeeze(-1),
            torch.sigmoid(self.head2(cls)).squeeze(-1),
            torch.sigmoid(self.head3(cls)),
        )


class _Pooler(nn.Module):
    """Matches the pooler.dense.* keys in model_best_m2.pt."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dense(x))


class _M2(nn.Module):
    """DeBERTa base + pooler + linear classifier for binary sentiment.

    Architecture reverse-engineered from model_best_m2.pt state dict keys:
      deberta.*           — AutoModel (DebertaV2Model)
      pooler.dense.*      — _Pooler linear layer
      classifier.weight/bias — output Linear (n_classes)

    Takes (sentence, aspect) pairs via the tokenizer's text/text_pair fields;
    returns raw logits — softmax is applied during inference.
    """

    def __init__(self, base_model: str, n_classes: int):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(base_model)
        hidden = self.deberta.config.hidden_size
        self.pooler = _Pooler(hidden)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ):
        kwargs: dict = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.deberta(**kwargs)
        pooled = self.pooler(out.last_hidden_state[:, 0, :])
        return self.classifier(pooled)


# Checkpoint helpers


def _load_state_dict(path: Path) -> dict:
    """Handle the three common save formats: raw state dict, checkpoint dict,
    or a fully serialised nn.Module."""
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in raw:
                return raw[key]
        return raw
    if isinstance(raw, nn.Module):
        # Model was saved whole — extract its weights
        return raw.state_dict()
    raise ValueError(f"Unrecognised checkpoint format at {path}: {type(raw)}")



# Inference service


class ABSAService:
    """Loads M1 and M2 at startup; exposes predict() and predict_batch()."""

    def __init__(self):
        self._load_m1()
        self._load_m2()
        self._m1.eval()
        self._m2.eval()
        logger.info("ABSAService ready.")

    # Loaders


    def _load_m1(self) -> None:
        logger.info("Loading M1 (RoBERTa+LoRA)…")
        m1_dir = BASE / "model1_multitask"
        cfg = json.loads((m1_dir / "config.json").read_text())

        # backbone path in config.json is relative to models_artifacts/, not model1_multitask/
        backbone_path = (BASE / cfg["backbone"]).resolve()

        self._aspect_classes: list[str] = cfg["aspect_classes"]

        thresh = json.loads((m1_dir / "thresholds.json").read_text())
        self._h1_thresh: float = thresh["h1"]
        self._h2_thresh: float = thresh["h2"]
        self._h3_thresh: dict[str, float] = thresh["h3"]

        self._m1 = _M1(
            backbone_path=backbone_path,
            n_aspects=cfg["n_aspects"],
            lora_r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_target=cfg["lora_target"],
        )
        sd = _load_state_dict(m1_dir / "model_best.pt")
        self._m1.load_state_dict(sd)
        self._m1_tok = AutoTokenizer.from_pretrained(str(m1_dir), local_files_only=True)
        logger.info("M1 loaded.")

    def _load_m2(self) -> None:
        logger.info("Loading M2 (DeBERTa sentiment)…")
        m2_dir = BASE / "model2_sentiment"
        cfg = json.loads((m2_dir / "m2_config.json").read_text())

        self._m2_threshold: float = json.loads(
            (m2_dir / "m2_threshold.json").read_text()
        )["conf_threshold_m2"]
        self._id2label: dict[str, str] = cfg["id2label"]

        self._m2 = _M2(base_model=cfg["base_model"], n_classes=cfg["n_classes"])
        sd = _load_state_dict(m2_dir / "model_best_m2.pt")
        self._m2.load_state_dict(sd)
        self._m2_tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        logger.info("M2 loaded.")


    # Inference


    @torch.no_grad()
    def predict(self, text: str) -> PredictResponse:
        # --- M1: relatedness + technicality + aspect detection ---
        enc1 = self._m1_tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        p_related, p_technical, p_aspects = self._m1(**enc1)

        is_related = p_related.item() >= self._h1_thresh
        is_technical = p_technical.item() >= self._h2_thresh

        if not (is_related and is_technical):
            return PredictResponse(
                text=text,
                is_related=is_related,
                is_technical=is_technical,
                sentiment_aspects=[],
                mentioned_aspects=[],
            )

        detected = [
            asp
            for asp, score in zip(self._aspect_classes, p_aspects[0].tolist())
            if score >= self._h3_thresh[asp]
        ]

        if not detected:
            return PredictResponse(
                text=text,
                is_related=True,
                is_technical=True,
                sentiment_aspects=[],
                mentioned_aspects=[],
            )

        # --- M2: all (sentence, aspect) pairs in one forward pass ---
        enc2 = self._m2_tok(
            [text] * len(detected),
            detected,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        logits = self._m2(**enc2)
        probs = torch.softmax(logits, dim=-1)

        sentiment_aspects: list[AspectSentiment] = []
        mentioned_aspects: list[str] = []

        for asp, prob_row in zip(detected, probs.tolist()):
            conf = max(prob_row)
            pred_idx = prob_row.index(conf)
            if conf >= self._m2_threshold:
                sentiment_aspects.append(
                    AspectSentiment(
                        aspect=asp,
                        sentiment=self._id2label[str(pred_idx)],
                        confidence=round(conf, 4),
                    )
                )
            else:
                mentioned_aspects.append(asp)

        return PredictResponse(
            text=text,
            is_related=True,
            is_technical=True,
            sentiment_aspects=sentiment_aspects,
            mentioned_aspects=mentioned_aspects,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictResponse]:
        return [self.predict(t) for t in texts]
