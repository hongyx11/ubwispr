import logging
import subprocess
import threading

import numpy as np

log = logging.getLogger(__name__)


class MicRecorder:
    """Mic capture via `arecord` subprocess — avoids needing libportaudio2."""

    def __init__(self, sample_rate: int = 16000, device: str = "default"):
        self.sample_rate = sample_rate
        self.device = device
        self._proc: subprocess.Popen | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stop_reader = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._proc is not None:
                return
            self._buffer = bytearray()
            self._stop_reader.clear()
            self._proc = subprocess.Popen(
                [
                    "arecord", "-q",
                    "-f", "S16_LE",
                    "-c", "1",
                    "-r", str(self.sample_rate),
                    "-t", "raw",
                    "-D", self.device,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while not self._stop_reader.is_set():
            data = stdout.read(4096)
            if not data:
                break
            with self._lock:
                self._buffer.extend(data)

    def stop(self) -> np.ndarray:
        with self._lock:
            proc = self._proc
            reader = self._reader
            self._proc = None
            self._reader = None
            self._stop_reader.set()
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
            except ProcessLookupError:
                pass
        if reader is not None:
            reader.join(timeout=1.0)
        with self._lock:
            raw = bytes(self._buffer)
            self._buffer = bytearray()
        if not raw:
            return np.zeros(0, dtype=np.float32)
        pcm = np.frombuffer(raw, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0
