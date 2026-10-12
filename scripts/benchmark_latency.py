"""Benchmark the configured ASR engine with cached weights and a fixed speech clip.

    conda run --no-capture-output -n ubwispr python scripts/benchmark_latency.py

No transcript is typed or stored. Timings include preprocessing, inference,
and decoding; microphone-stop timings are measured separately.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers.audio_utils import load_audio

from ubwispr.asr import AsrEngine
from ubwispr.config import load
from ubwispr.mic import MicRecorder


def summarize(samples: list[float]) -> dict:
    return {
        "median_ms": round(statistics.median(samples) * 1000, 2),
        "min_ms": round(min(samples) * 1000, 2),
        "max_ms": round(max(samples) * 1000, 2),
        "runs_ms": [round(value * 1000, 2) for value in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--idle-runs', type=int, default=3)
    parser.add_argument('--idle-gap-seconds', type=float, default=3)
    parser.add_argument('--durations', type=float, nargs='+', default=[3, 5, 10, 20])
    parser.add_argument('--audio-file', type=Path)
    parser.add_argument('--output', type=Path, default=Path('reports/latency.json'))
    parser.add_argument('--skip-mic', action='store_true')
    args = parser.parse_args()
    if args.runs < 1 or any(not np.isfinite(d) or d <= 0 for d in args.durations):
        parser.error('runs and durations must be positive')
    if args.idle_runs < 0 or not np.isfinite(args.idle_gap_seconds) or args.idle_gap_seconds < 0:
        parser.error('idle-runs and idle-gap-seconds must be nonnegative')
    logging.basicConfig(level=logging.WARNING)
    config = load()
    cuda = config.device != 'cpu'
    if cuda and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    if cuda:
        index = int(config.device.split(':')[1]) if ':' in config.device else 0
        torch.cuda.set_device(index)
    def sync():
        if cuda:
            torch.cuda.synchronize()

    # Loading a snapshot path avoids gated metadata requests for cached weights.
    snapshot = snapshot_download(config.model_id, local_files_only=True)
    if args.audio_file:
        audio_file = str(args.audio_file)
        source = str(args.audio_file)
    else:
        audio_file = hf_hub_download(
            repo_id='hf-internal-testing/dummy-audio-samples', repo_type='dataset',
            filename='bcn_weather.mp3',
        )
        source = 'hf-internal-testing/dummy-audio-samples/bcn_weather.mp3'
    audio = load_audio(audio_file, sampling_rate=config.sample_rate)
    if len(audio) == 0:
        parser.error('The audio sample is empty')
    source_seconds = len(audio) / config.sample_rate
    engine = AsrEngine(snapshot, device=config.device, dtype=config.dtype)
    print('Loading the cached model…', flush=True)
    t0 = time.perf_counter()
    engine.load()
    sync()
    load_seconds = time.perf_counter() - t0
    print(f'Model loaded in {load_seconds:.2f}s; source clip {source_seconds:.2f}s', flush=True)

    def clip(seconds):
        samples = int(seconds * config.sample_rate)
        return np.tile(audio, max(1, (samples + len(audio) - 1) // len(audio)))[:samples]

    def transcribe(utterance):
        sync()
        start = time.perf_counter()
        text = engine.transcribe(
            utterance, sample_rate=config.sample_rate, language=config.language,
            max_new_tokens=config.max_new_tokens,
        )
        sync()
        return time.perf_counter() - start, len(text)

    first_seconds = args.durations[0]
    first_time, first_chars = transcribe(clip(first_seconds))
    print(f'First inference ({first_seconds:g}s audio): {first_time * 1000:.1f}ms', flush=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    measurements = []
    for duration in args.durations:
        utterance = clip(duration)
        transcribe(utterance)  # One untimed warm-up per audio length.
        elapsed = []
        char_counts = []
        for _ in range(args.runs):
            seconds, chars = transcribe(utterance)
            elapsed.append(seconds)
            char_counts.append(chars)
        stats = summarize(elapsed)
        result = {
            'audio_seconds': len(utterance) / config.sample_rate,
            'warmed_recognition': stats,
            'realtime_factor': round(statistics.median(elapsed) / duration, 4),
            'output_character_counts': char_counts,
            'source_repeated': duration > source_seconds,
        }
        measurements.append(result)
        print(f'{duration:g}s audio: median {stats["median_ms"]:.1f}ms, '
              f'range {stats["min_ms"]:.1f}–{stats["max_ms"]:.1f}ms', flush=True)
        if args.idle_runs:
            idle_times = []
            idle_char_counts = []
            for _ in range(args.idle_runs):
                time.sleep(args.idle_gap_seconds)
                seconds, chars = transcribe(utterance)
                idle_times.append(seconds)
                idle_char_counts.append(chars)
            result['recognition_after_idle'] = summarize(idle_times)
            result['idle_output_character_counts'] = idle_char_counts
            print(f'  After {args.idle_gap_seconds:g}s idle: median '
                  f'{result["recognition_after_idle"]["median_ms"]:.1f}ms', flush=True)

    microphone = None
    if not args.skip_mic:
        recorder = MicRecorder(config.sample_rate, config.arecord_device, max_seconds=2)
        elapsed = []
        lengths = []
        try:
            for _ in range(5):
                recorder.start()
                time.sleep(0.7)
                start = time.perf_counter()
                captured = recorder.stop()
                elapsed.append(time.perf_counter() - start)
                lengths.append(len(captured))
            microphone = {'stop_and_drain': summarize(elapsed), 'captured_samples': lengths}
            print(f'Microphone stop/drain: median {microphone["stop_and_drain"]["median_ms"]:.1f}ms', flush=True)
        finally:
            recorder.stop()
    report = {
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_id': config.model_id,
        'dtype': config.dtype,
        'language': config.language,
        'max_new_tokens': config.max_new_tokens,
        'gpu': torch.cuda.get_device_name() if cuda else None,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'source': source,
        'source_audio_seconds': source_seconds,
        'runs_per_duration': args.runs,
        'idle_runs_per_duration': args.idle_runs,
        'idle_gap_seconds': args.idle_gap_seconds,
        'model_load_seconds': round(load_seconds, 3),
        'first_inference': {'audio_seconds': first_seconds, 'elapsed_ms': round(first_time * 1000, 2),
                            'output_characters': first_chars},
        'measurements': measurements,
        'microphone': microphone,
        'typing_settle_delay_ms': 50,
        'peak_allocated_vram_mib': round(torch.cuda.max_memory_allocated() / 1024**2, 1) if cuda else None,
        'peak_reserved_vram_mib': round(torch.cuda.max_memory_reserved() / 1024**2, 1) if cuda else None,
        'notes': [
            'Recognition includes preprocessing, GPU execution and text decoding, with CUDA synchronization.',
            'Longer samples repeat the source clip. This measures latency, not recognition accuracy.',
            'Microphone stop/drain is measured independently; queueing, notifications, key dispatch and text insertion are excluded.',
            'First inference and model loading are excluded from warmed statistics.',
            'Idle-gap runs keep the model loaded but leave the GPU unused between inferences.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Results saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
