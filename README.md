# TikTok Auto Video Generator (`tkvideobulk`)

## Guia de desenvolvimento

O [AGENTS.md](AGENTS.md) documenta o pipeline FFmpeg, seus invariantes de saída e a validação esperada.

Leia a [arquitetura técnica](docs/architecture.md) para o ciclo de análise, renderização, verificação e concorrência.

Batch-processing Python tool that automatically generates YouTube Shorts-ready vertical videos (1080×1920) by combining source videos with the highest-energy segments of music tracks.

---

## Features

- **High-Energy Audio Detection** — Scans each music file (RMS volume + onset/beat density) and picks the segment with the most impact to match the video duration.
- **Auto-Looping** — If a music track is shorter than the video, it loops seamlessly.
- **YouTube Shorts Format** — Crops and scales any input to **1080×1920** (9:16), limits clips to 3 minutes, and exports MP4 with H.264/AAC and fast start.
- **Clean Audio Pipeline** — Dynamic volume normalisation (`dynaudnorm`), smooth fade-in/out, consistent 48 kHz sample rate, AAC 192 kbps output.
- **Post-Render Verification** — Automatically checks each output is vertical, at most 3 minutes long, and has audio.
- **Two Energy Peaks** — Generates two clips for each video/music pair, using the two strongest non-overlapping music segments.
- **Safe Naming** — Output format: `videoname-00m15s-musicname_v1.mp4`. Numeric suffixes (`-01`, `-02`, …) prevent overwrites.

---

## Requirements

- **Python 3.8+**
- **FFmpeg** (with `ffmpeg` and `ffprobe` in your PATH)

### Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add bin/ to PATH
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Folder Structure

```
tkvideobulk/
├── main.py            # Main script
├── requirements.txt   # Python dependencies
├── README.md
├── in-video/          # ← Place source videos here (.mp4, .mov, .mkv, .avi)
├── in-wav/            # ← Place music files here (.wav, .mp3, .flac, .m4a)
└── out-video/         # → Generated TikTok videos appear here
```

Folders are created automatically if they don't exist.

---

## Usage

1. Drop your videos into `./in-video/`
2. Drop your music tracks into `./in-wav/`
3. Run:

```bash
python main.py
```

Each video is paired with **every** music track and produces two clips, so 3 videos × 2 songs × 2 peaks = 12 output files. Source files are preserved.

---

## Output Naming

```
videoname-duration-musicname_v1.mp4
```

**Example:**
| Input Video | Input Audio | Output |
|---|---|---|
| `Beach Volleyball.mov` (15s) | `Dark_Beat.mp3` | `beachvolleyball-00m15s-darkbeat_v1.mp4` |
| (second peak) | (same) | `beachvolleyball-00m15s-darkbeat_v2.mp4` |

Rules: lowercase, no spaces/special chars, duration in `XXmYYs` format.

---

## How It Works

1. **Probe** the video (resolution, duration) via `ffprobe`.
2. **Convert** the music to a temp 22050 Hz mono WAV.
3. **Analyse** the WAV — compute RMS energy and onset strength per frame, then slide a window matching the video duration to find the peak-energy section.
4. **Render** with FFmpeg — crop/scale video to 9:16, seek/loop the audio to the best segment, normalise volume, apply fades, encode H.264 + AAC.
5. **Verify** the output has a valid audio stream.
6. **Preserve** the source videos in `in-video/` after processing.
