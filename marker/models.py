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


def _get_multi_gpu_device_map(num_gpus: int) -> dict:
    """Distribute models across multiple GPUs.

    The pipeline is sequential (layout → detection → OCR → table → equation),
    so only one model runs at a time. Spreading models across GPUs maximizes
    available VRAM per model for inference.

    Priority: layout and recognition are the heaviest and get their own GPUs.
    """
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

    if num_gpus >= 4:
        # Each heavy model gets its own GPU
        return {
            "layout_model": devices[0],
            "recognition_model": devices[1],
            "detection_model": devices[2],
            "table_rec_model": devices[3],
            "ocr_error_model": devices[3],  # Lightest model, shares with table_rec
        }
    elif num_gpus == 3:
        return {
            "layout_model": devices[0],
            "recognition_model": devices[1],
            "detection_model": devices[2],
            "table_rec_model": devices[2],
            "ocr_error_model": devices[2],
        }
    else:  # num_gpus == 2
        return {
            "layout_model": devices[0],
            "recognition_model": devices[1],
            "detection_model": devices[0],
            "table_rec_model": devices[1],
            "ocr_error_model": devices[0],
        }


def create_model_dict(
    device=None, dtype=None, attention_implementation: str | None = None
) -> dict:
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = num_gpus > 1

    if use_multi_gpu:
        device_map = _get_multi_gpu_device_map(num_gpus)
    else:
        device_map = {
            "layout_model": device,
            "recognition_model": device,
            "detection_model": device,
            "table_rec_model": device,
            "ocr_error_model": device,
        }

    models = {}

    models["layout_model"] = LayoutPredictor(FoundationPredictor(
        checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT,
        attention_implementation=attention_implementation,
        device=device_map["layout_model"],
        dtype=dtype,
    ))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models["recognition_model"] = RecognitionPredictor(FoundationPredictor(
        checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT,
        attention_implementation=attention_implementation,
        device=device_map["recognition_model"],
        dtype=dtype,
    ))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models["table_rec_model"] = TableRecPredictor(
        device=device_map["table_rec_model"],
        dtype=dtype,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models["detection_model"] = DetectionPredictor(
        device=device_map["detection_model"],
        dtype=dtype,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models["ocr_error_model"] = OCRErrorPredictor(
        device=device_map["ocr_error_model"],
        dtype=dtype,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return models
