import os
import math
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    model_id: str = "CohereLabs/cohere-transcribe-03-2026"
    language: str = "en"
    hotkey: str = "alt_r"
    sample_rate: int = 16000
    min_record_seconds: float = 0.3
    max_record_seconds: float = 60.0
    device: str = "auto"
    dtype: str = "bfloat16"
    arecord_device: str = "default"
    max_new_tokens: int = 256
    notifications: bool = True

    @property
    def hotkey_label(self) -> str:
        if self.hotkey.startswith("f"):
            return self.hotkey.upper()
        if self.hotkey == "alt_gr":
            return "AltGr"
        key, side = self.hotkey.split("_")
        names = {"alt": "Alt", "ctrl": "Ctrl", "cmd": "Super", "shift": "Shift"}
        return f"{'Right' if side == 'r' else 'Left'} {names[key]}"

    def validate(self) -> None:
        for name in ("model_id", "language", "hotkey", "arecord_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.hotkey not in HOTKEYS:
            raise ValueError(f"Unsupported hotkey {self.hotkey!r}; choose from {', '.join(HOTKEYS)}")
        if not isinstance(self.device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("dtype must be bfloat16, float16, or float32")
        for name in ("sample_rate", "max_new_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_record_seconds", "max_record_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.min_record_seconds >= self.max_record_seconds:
            raise ValueError("min_record_seconds must be smaller than max_record_seconds")
        if type(self.notifications) is not bool:
            raise ValueError("notifications must be true or false")


HOTKEYS = (
    "alt_r", "alt_l", "alt_gr", "ctrl_r", "ctrl_l", "cmd_r", "cmd_l", "shift_r", "shift_l",
    *(f"f{i}" for i in range(1, 13)),
)


def _candidate_paths() -> list[Path]:
    env = os.environ.get("UBWISPR_CONFIG")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.append(Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "ubwispr" / "config.toml")
    paths.append(Path(__file__).resolve().parent.parent / "config.toml")
    return paths


def config_path(explicit: str | None = None) -> Path | None:
    requested = explicit or os.environ.get("UBWISPR_CONFIG")
    if requested:
        path = Path(requested).expanduser()
        if not path.is_file():
            raise ValueError(f"Configuration file does not exist: {path}")
        return path
    return next((path for path in _candidate_paths() if path.is_file()), None)


def load(explicit: str | None = None) -> Config:
    cfg = Config()
    allowed = {f.name for f in fields(Config)}
    path = config_path(explicit)
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        unknown = data.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
        for k, v in data.items():
            setattr(cfg, k, v)
    cfg.validate()
    return cfg
