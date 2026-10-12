import logging
import queue
import threading
import time

import numpy as np

from ubwispr.config import Config
from ubwispr.events import Status, StatusCallback
from ubwispr.mic import MicRecorder
from ubwispr.model_access import ModelAccessError

log = logging.getLogger("ubwispr")


def _hotkey_factory(*args):
    from ubwispr.hotkey import HoldHotkey
    return HoldHotkey(*args)


def _type_text(text: str) -> None:
    from ubwispr.hotkey import type_text
    type_text(text)


class UbWispr:
    def __init__(self, config: Config, on_status: StatusCallback | None = None, *,
                 asr=None, mic=None, typer=None, hotkey_factory=None):
        config.validate()
        self.config = config
        if asr is None:
            from ubwispr.asr import AsrEngine
            asr = AsrEngine(config.model_id, device=config.device, dtype=config.dtype)
        self.asr = asr
        self.mic = mic if mic is not None else MicRecorder(
            sample_rate=config.sample_rate, device=config.arecord_device,
            max_seconds=config.max_record_seconds,
        )
        self._on_status = on_status or (lambda status: None)
        self._typer = typer or _type_text
        self._hotkey_factory = hotkey_factory or _hotkey_factory
        self._jobs: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._recording = False
        self._busy = False
        self._paused = False
        self._loaded = False
        self._closed = False
        self._generation = 0
        self._record_id = 0
        self._timer: threading.Timer | None = None
        self._hotkey = None
        self._worker = None
        self._record_start_time = 0.0

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded

    def _status(self, state: str, message: str, notify: bool = False) -> None:
        if self._stopping.is_set() and state != "stopped":
            return
        log.info("%s", message)
        try:
            self._on_status(Status(state, message, notify))
        except Exception:
            log.exception("Could not update desktop status")

    def _ready(self) -> None:
        if self._paused:
            self._status("paused", "Dictation paused")
        else:
            self._status("ready", f"Ready — hold {self.config.hotkey_label} to dictate")

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            audio, generation = job
            failed = False
            try:
                if not self._stopping.is_set():
                    self._handle_audio(audio, generation)
            except Exception as exc:
                failed = True
                log.exception("Transcription job failed")
                self._status("error", f"Dictation failed: {exc}", notify=True)
            finally:
                with self._lock:
                    self._busy = False
                    if not failed or self._paused:
                        self._ready()

    def _handle_audio(self, audio: np.ndarray, generation: int) -> None:
        dur = len(audio) / self.config.sample_rate
        if dur < self.config.min_record_seconds:
            log.info("Skipped a %.2fs recording", dur)
            return
        with self._lock:
            if self._paused or self._stopping.is_set() or generation != self._generation:
                return
            self._status("transcribing", "Transcribing…")
        t0 = time.monotonic()
        text = self.asr.transcribe(
            audio, sample_rate=self.config.sample_rate, language=self.config.language,
            max_new_tokens=self.config.max_new_tokens,
        )
        # Transcripts are not written to logs or notifications.
        log.info("Transcribed %.2fs of audio in %.2fs", dur, time.monotonic() - t0)
        if not text:
            return
        # An ignored hotkey press during inference may still leave Alt held.
        # Never synthesize text until that physical key has been released.
        while self._hotkey is not None and not self._hotkey.wait_released():
            if self._stopping.is_set() or generation != self._generation:
                return
        with self._lock:
            if not self._stopping.is_set() and not self._paused and generation == self._generation:
                self._typer(text)

    def _on_press(self) -> None:
        with self._lock:
            if self._stopping.is_set() or self._paused or self._busy or not self._loaded:
                return
            try:
                self.mic.start()
            except Exception as exc:
                log.exception("Could not start microphone")
                self._status("error", f"Could not start microphone: {exc}", notify=True)
                return
            self._record_start_time = time.monotonic()
            self._record_id += 1
            self._recording = True
            self._busy = True
            self._status("recording", "Recording — release to transcribe")
            self._timer = threading.Timer(self.config.max_record_seconds, self._on_limit, args=(self._record_id,))
            self._timer.daemon = True
            self._timer.start()

    def _on_limit(self, record_id: int | None = None) -> None:
        self._finish_recording(at_limit=True, record_id=record_id)

    def _on_release(self) -> None:
        self._finish_recording()

    def _finish_recording(self, at_limit: bool = False, record_id: int | None = None) -> None:
        with self._lock:
            if not self._recording or (record_id is not None and record_id != self._record_id):
                return
            self._recording = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            try:
                audio = self.mic.stop()
            except Exception as exc:
                self._busy = False
                log.exception("Microphone capture failed")
                self._status("error", str(exc), notify=True)
                return
            log.info("Recorded %.2fs (%d samples)", time.monotonic() - self._record_start_time, len(audio))
            if at_limit:
                self._status("transcribing", "Recording limit reached — release the hotkey", notify=True)
            self._jobs.put_nowait((audio, self._generation))

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            if self._stopping.is_set():
                return
            self._paused = paused
            if paused:
                self._generation += 1
            if paused and self._recording:
                self._recording = False
                self._busy = False
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
                try:
                    self.mic.stop()  # Pausing cancels the current recording.
                except Exception:
                    log.exception("Could not stop microphone while pausing")
            if paused or not self._busy:
                self._ready()

    def stop(self, report: bool = True) -> None:
        # Set this before taking the lock so pending typing can observe shutdown.
        self._stopping.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._loaded = False
            self._generation += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._recording:
                self._recording = False
                try:
                    self.mic.stop()
                except Exception:
                    log.exception("Could not stop microphone")
            if self._hotkey is not None:
                self._hotkey.stop()
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass
            if report:
                self._status("stopped", "Ubuntu Wispr stopped")

    def run(self) -> None:
        failed = False
        try:
            self._status("loading", "Loading speech model…")
            self.asr.load()
            with self._lock:
                if self._stopping.is_set():
                    return
                self._loaded = True
                self._hotkey = self._hotkey_factory(self.config.hotkey, self._on_press, self._on_release)
                self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="ubwispr-asr")
                self._worker.start()
                self._status("ready", f"Ready — hold {self.config.hotkey_label} to dictate", notify=True)
            self._hotkey.run()
        except Exception as exc:
            failed = True
            if isinstance(exc, ModelAccessError):
                log.error("%s", exc)
            else:
                log.exception("Daemon failed")
            self._status("error", f"Could not start dictation: {exc}", notify=True)
            raise
        finally:
            self.stop(report=not failed)
