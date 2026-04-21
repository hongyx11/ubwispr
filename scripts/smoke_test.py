"""Smoke-test the Cohere ASR model end-to-end.

Downloads the model (first run only), runs the HF demo clip through it, and
prints the transcription. Use this to verify the weights + GPU work before
launching the daemon.

    uv run python scripts/smoke_test.py
"""

import logging
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_id = "CohereLabs/cohere-transcribe-03-2026"

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"processor loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"model loaded in {time.time() - t0:.1f}s on {model.device} (dtype={model.dtype})")

    audio_file = hf_hub_download(
        repo_id=model_id,
        filename="demo/voxpopuli_test_en_demo.wav",
    )
    audio = load_audio(audio_file, sampling_rate=16000)
    print(f"audio: {len(audio)} samples, {len(audio) / 16000:.2f}s")

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
    inputs = inputs.to(model.device, dtype=model.dtype)

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256)
    print(f"generated in {time.time() - t0:.2f}s")

    try:
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        text = decoded[0] if decoded else ""
    except Exception:
        text = processor.decode(outputs[0], skip_special_tokens=True)
    print("---")
    print(text)
    print("---")


if __name__ == "__main__":
    main()
