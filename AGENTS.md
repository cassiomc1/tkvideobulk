# tkvideobulk development guide

## Architecture

`main.py` discovers input media, selects high-energy non-overlapping audio windows, invokes FFmpeg, and verifies exported artifacts with FFprobe. Input, temporary, and output paths are part of the processing contract.

## Development

```bash
python -m unittest discover
python -m py_compile main.py
```

FFmpeg and FFprobe must be available on `PATH` for end-to-end runs.

## Invariants

- Never overwrite an existing output; reserve unique paths atomically.
- Remove incomplete output artifacts after a render failure.
- Verify every finished video is 9:16, no more than three minutes, and contains audio.
- Preserve source media. Do not commit input, output, temporary media, or generated caches.
