# Ubuntu Wispr

An open-source alternative to [Wispr Flow](https://wisprflow.ai/) for
Ubuntu / Linux. Hold-to-dictate speech-to-text powered by
[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
(2B-param Conformer ASR, 14 languages).

Hold **Right Alt**, speak, release — the transcript is typed into whichever
window has focus.

## License

MIT — see [LICENSE](LICENSE).

## Why not Fn?

The `Fn` key on laptops is handled by the embedded controller / firmware and
never reaches userspace on Linux. No application can bind to it. `alt_r` is
the default substitute; change it in the config.

## Requirements

- **NVIDIA GPU with ≥4 GB VRAM** (required — CPU is not supported). The
  model is ~4 GB in bfloat16 and fits on a 4 GB card with no headroom to
  spare; 6 GB+ is more comfortable. CUDA drivers must be installed.
- Linux with X11 (Wayland has a different global-hotkey story; not tested)
- Python 3.12 (auto-installed by `uv` if not present)
- `arecord` (ALSA utils — ships on every desktop distro)
- `libX11` + `libXtst` (already present on GNOME/X11 setups)
- ~5 GB free disk for model weights (cached under `~/.cache/huggingface`)

A Hugging Face token with access to the Cohere repo must be saved at
`~/.cache/huggingface/token` (standard location — `huggingface-cli login`
also writes there).

## Install

```sh
uv sync
```

## Smoke test (first run downloads ~4 GB)

```sh
uv run python scripts/smoke_test.py
```

Expected: model loads in ~30 s, inference runs in ~0.5 s, prints a
transcription of the bundled demo clip.

## Run the daemon

```sh
./run.sh
# or:
uv run python -m ubwispr
```

Loading the model takes ~30 s the first time (weights stream from disk to
GPU). Once you see `ready — hold alt_r to dictate`, hold Right Alt, speak,
release. The transcription will be typed at your cursor.

## Configuration

Defaults live in `ubwispr/config.py`. Override by copying
`config.example.toml` to one of (searched in this order):

1. `$UBWISPR_CONFIG`
2. `~/.config/ubwispr/config.toml`
3. `./config.toml` (repo root)

Useful fields:

| field                | default    | notes                                         |
| ---                  | ---        | ---                                           |
| `hotkey`             | `alt_r`    | `alt_r`/`alt_l`/`ctrl_r`/`cmd_r`/`f9` etc.    |
| `language`           | `en`       | 14 supported (en, fr, de, es, zh, ja, …)      |
| `device`             | `auto`     | `auto` resolves to `cuda` (only GPU is supported) |
| `dtype`              | `bfloat16` | `bfloat16`/`float16` (use `float16` on GPUs without bf16 support) |
| `arecord_device`     | `default`  | e.g. `plughw:1,4` to pick a specific mic      |
| `min_record_seconds` | `0.3`      | shorter holds are ignored (accidental taps)   |

## Autostart (GNOME)

Create `~/.config/autostart/ubwispr.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=ubwispr
Exec=/home/yuxihong/Documents/code/ubwispr/run.sh
Terminal=false
X-GNOME-Autostart-enabled=true
```

## Architecture

Single daemon process:

- **hotkey thread** — `pynput.keyboard.Listener`, edge-triggered on Right Alt
  down/up; debounces OS key-repeat.
- **mic** — `arecord` subprocess launched on press, killed on release; raw
  16-bit PCM piped back as a numpy float32 array. No `libportaudio2` needed.
- **worker thread** — consumes audio clips from a queue, runs ASR inference,
  types the result via `pynput.keyboard.Controller().type(...)`.
- **ASR engine** — HF Transformers `CohereAsrForConditionalGeneration`,
  loaded once at startup; `device_map="auto"` places it on CUDA.

Files:

```
ubwispr/
  __main__.py     # python -m ubwispr entry
  app.py          # daemon (wires everything together)
  asr.py          # model loading + transcribe()
  mic.py          # arecord subprocess wrapper
  hotkey.py       # pynput listener + typer
  config.py       # TOML config loader
scripts/
  smoke_test.py   # validate the model works end-to-end
```
