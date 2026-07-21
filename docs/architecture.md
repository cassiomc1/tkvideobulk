# tkvideobulk architecture and media-processing contract

## Pipeline

`main.py` discovers video files in `in-video/` and music files in `in-wav/`, pairs inputs, probes media with FFprobe, converts/analyzes audio, selects high-energy non-overlapping windows, invokes FFmpeg, and verifies each output before reporting it.

The tool creates its working directories when needed:

- `in-video/`: source video; preserved.
- `in-wav/`: source audio; preserved.
- `out-video/`: generated MP4 files.
- temporary files: created during conversion/analysis and removed after each operation.

## Render contract

Outputs are vertical 1080×1920 H.264/AAC MP4 files, capped at three minutes. The audio path applies the selected source window, looping only when needed, plus its normalization and fade processing. The final verifier must confirm video dimensions, duration, and an audio stream before treating an output as successful.

`unique_output_path` reserves output names atomically with exclusive creation. Never replace that with a check-then-write sequence: concurrent workers must not overwrite or share a target path. On render failure, remove the reserved incomplete artifact while preserving original media.

## Resource behavior

The process limits workers and encoder pressure, and optionally pauses the queue when `psutil` reports excessive CPU load. The monitor is a throughput safeguard, not a substitute for capacity planning. Keep its messages and report writes synchronized when changing concurrency.

## Local validation

```bash
python -m unittest discover
python -m py_compile main.py
python main.py
```

End-to-end execution requires `ffmpeg` and `ffprobe` on `PATH`. Validate a small fixture set before processing a large library, then inspect at least one produced file with FFprobe and playback. Do not commit input media, output media, temporary files, or `report.txt`.
