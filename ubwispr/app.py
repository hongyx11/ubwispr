import logging
import queue
import threading
import time

import numpy as np

from ubwispr.asr import AsrEngine
from ubwispr.config import Config, load as load_config
from ubwispr.hotkey import HoldHotkey, type_text
from ubwispr.mic import MicRecorder

log = logging.getLogger("ubwispr")


class UbWispr:
    def __init__(self, config: Config):
        self.config = config
        self.asr = AsrEngine(config.model_id, device=config.device, dtype=config.dtype)
        self.mic = MicRecorder(sample_rate=config.sample_rate, device=config.arecord_device)
        self._jobs: queue.Queue = queue.Queue()
        self._record_start_time = 0.0

    def _worker_loop(self) -> None:
        while True:
            audio = self._jobs.get()
            if audio is None:
                return
            try:
                self._handle_audio(audio)
            except Exception:
                log.exception("Transcription job failed")

    def _handle_audio(self, audio: np.ndarray) -> None:
        dur = len(audio) / self.config.sample_rate
        if dur < self.config.min_record_seconds:
            log.info("skip: %.2fs clip below min_record_seconds=%.2f", dur, self.config.min_record_seconds)
            return
        if dur > self.config.max_record_seconds:
            log.warning("clip %.2fs exceeds max; truncating", dur)
            audio = audio[: int(self.config.max_record_seconds * self.config.sample_rate)]
        log.info("transcribing %.2fs...", dur)
        t0 = time.time()
        text = self.asr.transcribe(
            audio,
            sample_rate=self.config.sample_rate,
            language=self.config.language,
            max_new_tokens=self.config.max_new_tokens,
        )
        elapsed = time.time() - t0
        log.info("transcribed in %.2fs: %r", elapsed, text)
        if text:
            type_text(text)

    def _on_press(self) -> None:
        self._record_start_time = time.time()
        self.mic.start()
        log.info("▶ recording...")

    def _on_release(self) -> None:
        audio = self.mic.stop()
        dur = time.time() - self._record_start_time
        log.info("■ stopped after %.2fs (%d samples)", dur, len(audio))
        self._jobs.put(audio)

    def run(self) -> None:
        log.info("loading ASR model (%s)...", self.config.model_id)
        self.asr.load()
        log.info("ready — hold %s to dictate (release to transcribe)", self.config.hotkey)

        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()

        hk = HoldHotkey(self.config.hotkey, self._on_press, self._on_release)
        hk.run()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("transformers", "httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config()
    try:
        UbWispr(cfg).run()
    except KeyboardInterrupt:
        log.info("bye.")
