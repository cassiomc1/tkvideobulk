# TikTok Auto Video Generator (`tkvideobulk`)

Batch-processing Python tool that automatically generates TikTok-style vertical videos (1080×1920) by combining source videos with the highest-energy segments of music tracks.

---

## Features

- **High-Energy Audio Detection** — Scans each music file (RMS volume + onset/beat density) and picks the segment with the most impact to match the video duration.
- **Auto-Looping** — If a music track is shorter than the video, it loops seamlessly.
- **TikTok Vertical Format** — Crops and scales any input to **1080×1920** (9:16). Horizontal videos are center-cropped; vertical ones scaled perfectly.
- **Clean Audio Pipeline** — Dynamic volume normalisation (`dynaudnorm`), smooth fade-in/out, consistent 44.1 kHz sample rate, AAC 192 kbps output.
- **Post-Render Verification** — Automatically checks each output file has a valid audio stream before marking it as successful.
- **Auto-Cleanup** — After all outputs for a source video succeed, the original is removed from `in-video/` so it won't be re-processed on the next run.
- **Safe Naming** — Output format: `videoname-00m15s-musicname.mp4`. Numeric suffixes (`-01`, `-02`, …) prevent overwrites.

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

Each video is paired with **every** music track, so 3 videos × 2 songs = 6 output files.

After successful processing, source videos are automatically removed from `in-video/`.

---

## Output Naming

```
videoname-duration-musicname.mp4
```

**Example:**
| Input Video | Input Audio | Output |
|---|---|---|
| `Beach Volleyball.mov` (15s) | `Dark_Beat.mp3` | `beachvolleyball-00m15s-darkbeat.mp4` |
| (same, run again) | (same) | `beachvolleyball-00m15s-darkbeat-01.mp4` |

Rules: lowercase, no spaces/special chars, duration in `XXmYYs` format.

---

## How It Works

1. **Probe** the video (resolution, duration) via `ffprobe`.
2. **Convert** the music to a temp 22050 Hz mono WAV.
3. **Analyse** the WAV — compute RMS energy and onset strength per frame, then slide a window matching the video duration to find the peak-energy section.
4. **Render** with FFmpeg — crop/scale video to 9:16, seek/loop the audio to the best segment, normalise volume, apply fades, encode H.264 + AAC.
5. **Verify** the output has a valid audio stream.
6. **Clean up** source videos from `in-video/` after all outputs succeed.
