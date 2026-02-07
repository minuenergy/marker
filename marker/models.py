import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for an op, which is not supported on MPS
)

import torch

from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor
from surya.settings import settings as surya_settings


def create_model_dict(
    device=None, dtype=None, attention_implementation: str | None = None
) -> dict:
    models = {}

    models["layout_model"] = LayoutPredictor(FoundationPredictor(checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT, attention_implementation=attention_implementation, device=device, dtype=dtype))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models["recognition_model"] = RecognitionPredictor(FoundationPredictor(checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT, attention_implementation=attention_implementation, device=device, dtype=dtype))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models["table_rec_model"] = TableRecPredictor(device=device, dtype=dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models["detection_model"] = DetectionPredictor(device=device, dtype=dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models["ocr_error_model"] = OCRErrorPredictor(device=device, dtype=dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return models
