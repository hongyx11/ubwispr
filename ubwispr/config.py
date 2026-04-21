import os
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


def _candidate_paths() -> list[Path]:
    env = os.environ.get("UBWISPR_CONFIG")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.append(Path.home() / ".config" / "ubwispr" / "config.toml")
    paths.append(Path(__file__).resolve().parent.parent / "config.toml")
    return paths


def load() -> Config:
    cfg = Config()
    allowed = {f.name for f in fields(Config)}
    for path in _candidate_paths():
        if path.is_file():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            for k, v in data.items():
                if k in allowed:
                    setattr(cfg, k, v)
            break
    return cfg
