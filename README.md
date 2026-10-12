# Ubuntu Wispr

Local hold-to-dictate speech-to-text for Ubuntu / Linux, powered by
[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

Hold **Right Alt**, speak, release — the transcript is typed into the focused
window. The model stays loaded between recordings. Audio is processed locally;
transcripts are not saved or included in logs or notifications.

## Desktop controls

The tray icon shows loading, ready, recording, transcribing, paused, and error
states. Its menu provides **Show status**, **Pause dictation**, **Open
configuration**, a notification toggle, and **Quit**. If your desktop does not
provide a system tray, a small status window provides the same core controls.

Pausing cancels an active recording and suppresses pending text insertion.
Recordings stop automatically at the configured duration limit. If you still
hold the hotkey, text insertion waits for release. A new recording is accepted
once the previous transcription finishes.

Text is inserted into the window focused when transcription finishes. Keep your
intended editor focused until insertion completes.

## Requirements

- Linux **X11** desktop session. Wayland is currently rejected at startup because
  global hotkeys and text insertion are not implemented for it.
- Miniconda / Conda. `environment.yml` creates a dedicated Python 3.12 environment.
- NVIDIA GPU with CUDA drivers for the default configuration. Model weights
  alone are roughly 4 GB in bfloat16; leave room for inference overhead.
  CPU placement is available but experimental; use
  `dtype = "float32"` for that path.
- ALSA `arecord` (`alsa-utils`), `libX11`, and `libXtst`.
- Desktop notifications use `notify-send` (`libnotify-bin`), when available.
- The optional Qt tray interface needs the desktop's Qt/XCB system libraries.
  On Ubuntu, missing-library errors may require `libxcb-cursor0` or
  `libxkbcommon-x11-0`. Use `--no-tray` to run without Qt.
- Several GB of disk space for model weights in the Hugging Face cache.
  The default Cohere model requires Hugging Face authentication and accepting
  its access conditions; see the model setup below.

## Model access (required on first setup)

1. Sign into [the Cohere model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
   and accept its access conditions using your own account.
2. Create a [Hugging Face read token](https://huggingface.co/settings/tokens)
   for that same account. If you use a fine-grained token, grant read access to
   this gated repository.
3. After creating the environment below, run
   `conda run --no-capture-output -n ubwispr hf auth login` and enter the token at its
   interactive prompt. This saves it in Hugging Face's local token store; do
   not put it in your app config or commit it.
4. Run `./run.sh --check`, then `./run.sh`.

Browser login alone does not authenticate model downloads. If access is still
denied after CLI login, check that the account has model access and the token
has the necessary permissions. `conda run -n ubwispr hf auth whoami` checks the signed-in
account. The app also honors `HF_TOKEN` and Hugging Face's `HF_HOME` setting.

## Install and run

```sh
~/miniconda3/bin/conda env create -f environment.yml
./run.sh --check
./run.sh
```

The environment installs the package versions in `requirements.lock.txt` and
the app's desktop extra. The app is installed in editable mode, so code changes
take effect on the next launch. Python packages are managed by pip **inside**
the Conda environment; system Python and the Conda base environment are not used
to run dictation.

`run.sh` finds `~/miniconda3`, activates the `ubwispr` environment, and runs the
app. You do not need to activate Conda in your terminal first. For a different
installation, set `UBWISPR_CONDA_EXE` to its `bin/conda` executable. To select a
different app environment, set `UBWISPR_CONDA_ENV` (default: `ubwispr`).

If you update dependencies, update their pins in `requirements.lock.txt` and run:

```sh
~/miniconda3/bin/conda env update -f environment.yml
```

`--check` checks configuration, the X11 connection, ALSA availability, CUDA,
GPU dtype support, and Hugging Face model access with a metadata request. It
does **not** download model files, load weights, capture audio, or run inference.
Run it from your desktop terminal; an SSH/TTY session without an X11 display
will fail the desktop check. Its model-access check needs a network connection;
cached inference can still work offline without running that check.

The first actual launch downloads model weights. Once the ready notification
appears, hold Right Alt to dictate. Startup errors are logged and also sent as
desktop notifications when enabled. Only one daemon can run per user.

For the terminal interface:

```sh
./run.sh --no-tray
```

Other options:

```sh
./run.sh --config /path/to/config.toml
./run.sh --no-notifications
./run.sh --help
```

## Configuration

Use **Open configuration** in the tray/status window to open your current TOML
file, creating a default user config if none exists. Edit the file and restart
Ubuntu Wispr to apply changes. The tray notification toggle applies to the
current session; change the file to persist it.

Alternatively, copy `config.example.toml` to
`~/.config/ubwispr/config.toml`. Configuration is selected in this order:

1. `--config PATH`, if supplied.
2. `$UBWISPR_CONFIG`, if set.
3. `$XDG_CONFIG_HOME/ubwispr/config.toml` (defaults to `~/.config/ubwispr/config.toml`).
4. `config.toml` in the repository root.

An explicitly requested missing file, invalid value, or unknown field is
reported as an error instead of silently falling back to defaults.

| Field | Default | Notes |
| --- | --- | --- |
| `hotkey` | `alt_r` | Single modifier or function key, such as `ctrl_r` or `f9` |
| `language` | `en` | Input language passed to the Cohere processor |
| `device` | `auto` | Uses CUDA; `cuda:N` selects a GPU; `cpu` is experimental |
| `dtype` | `bfloat16` | `float16` for older GPUs; `float32` also available |
| `arecord_device` | `default` | ALSA device, e.g. `plughw:1,4` |
| `sample_rate` | `16000` | Mono audio sample rate |
| `min_record_seconds` | `0.3` | Shorter clips are ignored |
| `max_record_seconds` | `60.0` | Automatically stops capture; bounds audio memory |
| `max_new_tokens` | `256` | Model decoding limit |
| `notifications` | `true` | Ready, errors, and recording-limit notifications |

Laptop Fn keys are commonly handled by firmware rather than exposed as normal
key events. Right Alt is the default substitute.

## Autostart

After installing dependencies, create `~/.config/autostart/ubwispr.desktop`.
Replace the `Exec` value below with the absolute path to your checkout's
`run.sh`; quote the path if it contains spaces.

```ini
[Desktop Entry]
Type=Application
Name=Ubuntu Wispr
Exec=/absolute/path/to/ubwispr/run.sh
Icon=audio-input-microphone
Terminal=false
X-GNOME-Autostart-enabled=true
```

The launcher discovers Miniconda even when the desktop's PATH does not include
Conda. GNOME setups without tray support display the status window.

## Validation

Run the headless regression suite without downloading weights:

```sh
~/miniconda3/bin/conda run --no-capture-output -n ubwispr python -m unittest discover -s tests -v
```

The tests cover configuration, final audio-block capture, bounded recording,
ALSA failures, duplicate hotkey edges, pause/cancellation, inference failures,
shutdown, instance locking, CLI diagnostics, and Qt controls/signals offscreen.
They use fake microphone/model objects; they do not verify real speech accuracy
or global key delivery.

For real model inference, the smoke test downloads the weights and a demo clip:

```sh
~/miniconda3/bin/conda run --no-capture-output -n ubwispr python scripts/smoke_test.py
```

A full desktop check still requires an X11 session, a working microphone, and
an editor: dictate a sentence, check its ending, pause/resume, then quit during
a recording and verify capture stops.

To benchmark latency, quit the app to free GPU memory, then run:

```sh
~/miniconda3/bin/conda run --no-capture-output -n ubwispr python scripts/benchmark_latency.py
```

The benchmark uses cached model weights and a public speech clip, measures
preprocessing/inference/decoding across several clip lengths, compares repeated
calls with calls after an idle gap, and measures microphone stop/drain overhead.
It saves results to `reports/latency.json` and does not type or store transcripts.
Use `--audio-file /path/to/clip.wav` for your own reference, or `--skip-mic`
to avoid microphone capture. Relaunch the app with `./run.sh` afterward.

## Architecture

- **Qt main thread** — optional PySide6 tray and status window. Worker updates
  reach widgets through Qt signals.
- **Hotkey listener** — `pynput` handles press/release and repeated presses.
- **Microphone reader** — `arecord` supplies 16-bit mono PCM, converted to NumPy
  float32. On release, the process terminates and its pipe is drained to EOF.
- **ASR worker** — consumes recordings, runs inference, and inserts text after
  checking cancellation and hotkey release.
- **ASR engine** — PyTorch, Hugging Face Transformers
  `CohereAsrForConditionalGeneration`, and Accelerate; loaded once at startup.
- **Startup** — validates TOML, checks runtime prerequisites, and takes a
  per-user process lock before loading weights.

No cloud transcription API, web server, or database is used.

## License

MIT — see [LICENSE](LICENSE).
