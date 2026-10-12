"""Desktop preflight and one-instance protection; no model downloads."""
import fcntl
import os
from pathlib import Path
import shutil

from ubwispr.config import Config


def check(config: Config) -> list[tuple[bool, str]]:
    checks = []
    has_display = bool(os.environ.get("DISPLAY"))
    wayland = os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))
    checks.append((has_display and not wayland, "X11 desktop session" if has_display and not wayland
                   else "An X11 desktop session is required; Wayland is not supported yet"))
    if has_display and not wayland:
        from Xlib.display import Display
        try:
            display = Display()
            display.close()
        except Exception:
            checks.append((False, "Cannot connect to X11; launch from your desktop session"))
        else:
            checks.append((True, "X11 connection available"))
    checks.append((bool(shutil.which("arecord")), "ALSA arecord installed" if shutil.which("arecord")
                   else "Install alsa-utils to provide arecord"))
    import torch
    if config.device == "cpu":
        checks.append((True, "CPU inference selected (experimental; use dtype = float32)"))
    else:
        available = torch.cuda.is_available()
        checks.append((available, "CUDA available" if available else "CUDA unavailable; check NVIDIA drivers and PyTorch"))
        if available:
            index = int(config.device.split(":")[1]) if ":" in config.device else 0
            valid_index = index < torch.cuda.device_count()
            checks.append((valid_index, f"CUDA device {index}" if valid_index else f"CUDA device {index} does not exist"))
            if valid_index and config.dtype == "bfloat16":
                with torch.cuda.device(index):
                    bf16 = torch.cuda.is_bf16_supported()
                checks.append((bf16, "bfloat16 supported" if bf16 else "Set dtype = float16; this GPU does not support bfloat16"))
    return checks


class InstanceLock:
    def __init__(self):
        # Use a stable per-user path even when different launchers have different
        # XDG_RUNTIME_DIR settings. Never unlink an active flock file.
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        self.path = cache / "ubwispr" / "daemon.lock"
        self._file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a")
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.close()
            self._file = None
            raise RuntimeError("Ubuntu Wispr is already running") from None
        return self

    def __exit__(self, *_):
        if self._file is not None:
            self._file.close()
            self._file = None
