import logging

import numpy as np
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

log = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class AsrEngine:
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "bfloat16"):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = _DTYPES.get(dtype, torch.bfloat16)
        self.processor = None
        self.model = None

    def load(self) -> None:
        log.info("Loading processor: %s", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        device_map = self.device if self.device != "cpu" else None
        log.info(
            "Loading model: %s (device_map=%s, dtype=%s)",
            self.model_id, device_map, self.torch_dtype,
        )
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=device_map,
            torch_dtype=self.torch_dtype,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        log.info("Model ready on %s", self.model.device)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
        max_new_tokens: int = 256,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("AsrEngine.load() must be called first")
        if audio.size == 0:
            return ""
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language=language,
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        try:
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            text = decoded[0] if decoded else ""
        except Exception:
            text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return text.strip()
