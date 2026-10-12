import logging
import math
import os
import re
import signal
import subprocess
import tempfile
import threading

import numpy as np

log = logging.getLogger(__name__)


class MicRecorder:
    """Capture bounded PCM audio and drain the pipe completely on release."""

    def __init__(self, sample_rate: int = 16000, device: str = "default", max_seconds: float = 60.0):
        self.sample_rate = sample_rate
        self.device = device
        self.max_seconds = max_seconds
        self._max_bytes = int(sample_rate * max_seconds) * 2
        self._proc: subprocess.Popen | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stderr = None
        self._read_error: Exception | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._proc is not None:
                return
            self._buffer = bytearray()
            self._read_error = None
            errors = tempfile.TemporaryFile()
            try:
                proc = subprocess.Popen(
                    [
                        "arecord", "-q", "-f", "S16_LE", "-c", "1",
                        "-r", str(self.sample_rate), "-t", "raw", "-D", self.device,
                        "-d", str(math.ceil(self.max_seconds)),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=errors,
                    env={**os.environ, "LC_ALL": "C"},
                )
            except Exception:
                errors.close()
                raise
            self._proc = proc
            self._stderr = errors
            self._reader = threading.Thread(target=self._read_loop, args=(proc,), daemon=True)
            self._reader.start()

    def _read_loop(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        try:
            while data := proc.stdout.read1(4096):
                with self._lock:
                    remaining = self._max_bytes - len(self._buffer)
                    self._buffer.extend(data[:remaining])
        except Exception as exc:
            self._read_error = exc

    def stop(self) -> np.ndarray:
        with self._lifecycle_lock:
            proc = self._proc
            if proc is None:
                return np.zeros(0, dtype=np.float32)
            try:
                requested_stop = proc.poll() is None
                if requested_stop:
                    proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
                # Read until EOF, including the final short block. Stopping the reader
                # before terminating arecord loses the last syllable of short clips.
                if self._reader is not None:
                    self._reader.join(timeout=2.0)
                    if self._reader.is_alive():
                        raise RuntimeError("Microphone reader did not stop")
                self._stderr.seek(0)
                detail = self._stderr.read(4096).decode(errors="replace").strip()
                # arecord 1.2.9 can exit with 1 after our SIGTERM interrupts
                # its ALSA read. Quiet mode suppresses the signal message, but
                # an interrupted PCM read can still report EINTR. Accept only
                # those diagnostics after our stop; device errors remain errors.
                stop_diagnostics = not detail or all(
                    re.fullmatch(r"arecord: pcm_read:\d+: read error: Interrupted system call", line)
                    for line in detail.splitlines()
                )
                expected_stop = requested_stop and stop_diagnostics and proc.returncode in (
                    1, -signal.SIGTERM, -signal.SIGKILL,
                )
                if proc.returncode != 0 and not expected_stop:
                    raise RuntimeError(f"Microphone capture failed: {detail or 'check your ALSA device'}")
                if self._read_error is not None:
                    raise RuntimeError("Could not read microphone audio") from self._read_error
                with self._lock:
                    raw = bytes(self._buffer)
                # Ignore an incomplete trailing sample if a device exits mid-write.
                pcm = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2")
                return pcm.astype(np.float32) / 32768.0
            finally:
                if proc.stdout is not None:
                    proc.stdout.close()
                self._stderr.close()
                self._proc = None
                self._reader = None
                self._stderr = None
                with self._lock:
                    self._buffer = bytearray()
