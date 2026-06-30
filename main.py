#!/usr/bin/env python3
"""
TikTok Auto Video Generator (tkvideobulk)

Batch-processes videos from ./in-video, pairs each with every music track
from ./in-wav, extracts the highest-energy segment, and renders
vertical 1080x1920 TikTok-ready clips into ./out-video.
"""

import os
import sys
import re
import json
import shutil
import tempfile
import warnings
import subprocess
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Directories ──────────────────────────────────────────────────────────────
IN_VIDEO_DIR = "./in-video"
IN_WAV_DIR = "./in-wav"
OUT_VIDEO_DIR = "./out-video"
REPORT_FILE = "./report.txt"

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

# ── Logging helpers ──────────────────────────────────────────────────────────
import threading
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor

_RST = "\033[0m"
_INF = "\033[94m"
_OK  = "\033[92m"
_WRN = "\033[93m"
_ERR = "\033[91m"
_DIM = "\033[90m"

print_lock = threading.Lock()
report_lock = threading.Lock()

def log_info(msg):
    with print_lock:
        print(f"{_INF}[INFO]{_RST} {msg}")
def log_success(msg):
    with print_lock:
        print(f"{_OK}[OK]{_RST} {msg}")
def log_warning(msg):
    with print_lock:
        print(f"{_WRN}[WARN]{_RST} {msg}")
def log_error(msg):
    with print_lock:
        print(f"{_ERR}[ERROR]{_RST} {msg}")

# ── Prerequisite checks ─────────────────────────────────────────────────────
def validate_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            log_error(f"{tool} not found in PATH. Install FFmpeg first.")
            sys.exit(1)
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        log_info(f"FFmpeg: {out.stdout.splitlines()[0]}")
    except Exception:
        pass

def ensure_dirs():
    for d in (IN_VIDEO_DIR, IN_WAV_DIR, OUT_VIDEO_DIR):
        os.makedirs(d, exist_ok=True)

# ── Media probing ────────────────────────────────────────────────────────────
def get_video_info(path):
    """Returns dict with width, height, duration (seconds)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration:format=duration",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    w = h = dur = None
    if info.get("streams"):
        s = info["streams"][0]
        w, h, dur = s.get("width"), s.get("height"), s.get("duration")
    if dur is None and info.get("format"):
        dur = info["format"].get("duration")
    if w is None or h is None or dur is None:
        raise ValueError("Could not read width/height/duration from video metadata.")
    return {"width": int(w), "height": int(h), "duration": float(dur)}


def get_audio_duration(path):
    """Returns audio duration in seconds via ffprobe, or None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return None

# ── Filename helpers ─────────────────────────────────────────────────────────
def sanitize_name(filename):
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    s = re.sub(r"[^a-z0-9]", "", base)
    return s or "file"

def format_duration(secs):
    t = int(round(secs))
    return f"{t // 60:02d}m{t % 60:02d}s"

def unique_output_path(video_name, dur_str, music_name):
    base = f"{video_name}-{dur_str}-{music_name}"
    candidate = os.path.join(OUT_VIDEO_DIR, f"{base}.mp4")
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate = os.path.join(OUT_VIDEO_DIR, f"{base}-{i:02d}.mp4")
        if not os.path.exists(candidate):
            return candidate
        i += 1

# ── Audio conversion ─────────────────────────────────────────────────────────
def convert_to_wav(src, dst):
    """Converts any audio to mono 22050 Hz 16-bit WAV via ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "22050", "-ac", "1",
         "-c:a", "pcm_s16le", dst],
        capture_output=True, check=True,
    )

# ── Energy analysis ──────────────────────────────────────────────────────────
def _analyze_wav_librosa(path, seg_dur, peak_index=0):
    """Finds highest-energy window using librosa RMS + onset strength."""
    import librosa
    import numpy as np

    y, sr = librosa.load(path, sr=22050, mono=True)
    total = len(y) / sr
    if total <= seg_dur:
        return 0.0, True

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    n = min(len(rms), len(onset))
    rms, onset = rms[:n], onset[:n]

    def norm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if (hi - lo) > 1e-6 else np.zeros_like(a)

    score = 0.5 * norm(rms) + 0.5 * norm(onset)
    win = max(1, int(round(seg_dur * sr / hop)))
    if win >= len(score):
        return 0.0, False

    sums = np.convolve(score, np.ones(win), mode="valid")
    best1 = int(np.argmax(sums))
    if peak_index == 0:
        best = best1
    else:
        sums_copy = sums.copy()
        start_idx = max(0, best1 - win)
        end_idx = min(len(sums), best1 + win)
        sums_copy[start_idx:end_idx] = -1e9
        if np.max(sums_copy) < -1e8:
            sums_copy = sums.copy()
            start_idx = max(0, best1 - win // 2)
            end_idx = min(len(sums), best1 + win // 2)
            sums_copy[start_idx:end_idx] = -1e9
        if np.max(sums_copy) < -1e8:
            best = (best1 + len(sums) // 2) % len(sums)
        else:
            best = int(np.argmax(sums_copy))

    t = best * hop / sr
    if t + seg_dur > total:
        t = max(0.0, total - seg_dur)
    return t, False


def _analyze_wav_numpy(path, seg_dur, peak_index=0):
    """Fallback energy analyzer using raw scipy.io.wavfile + numpy."""
    import numpy as np
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    # normalise to float32
    dtype_map = {
        "int16": 32768.0, "int32": 2147483648.0, "uint8": 128.0,
    }
    divisor = dtype_map.get(str(data.dtype))
    if divisor:
        data = data.astype(np.float32) / divisor
    else:
        data = data.astype(np.float32)

    total = len(data) / sr
    if total <= seg_dur:
        return 0.0, True

    hop, frame_len = 512, 2048
    nf = (len(data) - frame_len) // hop + 1
    if nf <= 0:
        return 0.0, False

    rms = np.array(
        [np.sqrt(np.mean(data[i * hop : i * hop + frame_len] ** 2) + 1e-10)
         for i in range(nf)],
        dtype=np.float32,
    )
    onset = np.zeros(nf, np.float32)
    onset[1:] = np.maximum(0.0, np.diff(rms))

    def norm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if (hi - lo) > 1e-6 else np.zeros_like(a)

    score = 0.5 * norm(rms) + 0.5 * norm(onset)
    win = max(1, int(round(seg_dur * sr / hop)))
    if win >= len(score):
        return 0.0, False

    sums = np.convolve(score, np.ones(win), mode="valid")
    best1 = int(np.argmax(sums))
    if peak_index == 0:
        best = best1
    else:
        sums_copy = sums.copy()
        start_idx = max(0, best1 - win)
        end_idx = min(len(sums), best1 + win)
        sums_copy[start_idx:end_idx] = -1e9
        if np.max(sums_copy) < -1e8:
            sums_copy = sums.copy()
            start_idx = max(0, best1 - win // 2)
            end_idx = min(len(sums), best1 + win // 2)
            sums_copy[start_idx:end_idx] = -1e9
        if np.max(sums_copy) < -1e8:
            best = (best1 + len(sums) // 2) % len(sums)
        else:
            best = int(np.argmax(sums_copy))

    t = best * hop / sr
    if t + seg_dur > total:
        t = max(0.0, total - seg_dur)
    return t, False


def analyze_audio(audio_path, seg_dur, peak_index=0):
    """
    Converts audio to a temp WAV first (guarantees format compatibility),
    then runs energy analysis. Returns (start_time, should_loop).
    """
    # Quick duration check — if shorter, we know we need looping already
    dur = get_audio_duration(audio_path)
    if dur is not None and dur <= seg_dur:
        return 0.0, True

    # Thread-safe temp file naming by including thread identifier
    tid = threading.get_ident()
    tmp = os.path.join(tempfile.gettempdir(), f"tkvb_{os.getpid()}_{tid}_{sanitize_name(audio_path)}.wav")
    try:
        convert_to_wav(audio_path, tmp)

        # Try librosa first, fall back to numpy
        try:
            return _analyze_wav_librosa(tmp, seg_dur, peak_index=peak_index)
        except Exception as e:
            log_warning(f"Librosa failed ({e}), falling back to NumPy analyser.")
            try:
                return _analyze_wav_numpy(tmp, seg_dur, peak_index=peak_index)
            except Exception as e2:
                log_error(f"NumPy analyser also failed: {e2}")
                return 0.0, False
    except Exception as conv_err:
        log_error(f"Audio conversion failed: {conv_err}")
        return 0.0, False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

# ── Video filter ─────────────────────────────────────────────────────────────
def _video_filter(w, h):
    """Returns ffmpeg filter chain to crop/scale to 1080x1920 (9:16)."""
    ar = w / h
    target = 9.0 / 16.0
    if abs(ar - target) < 0.01:
        return "scale=1080:1920"
    elif ar > target:
        # wider → crop width
        return (
            "crop=w='trunc(ih*9/16/2)*2':h='trunc(ih/2)*2'"
            ":x='(iw-ow)/2':y=0,scale=1080:1920"
        )
    else:
        # taller → crop height
        return (
            "crop=w='trunc(iw/2)*2':h='trunc(iw*16/9/2)*2'"
            ":x=0:y='(ih-oh)/2',scale=1080:1920"
        )

# ── FFmpeg render ────────────────────────────────────────────────────────────
def render(video_path, audio_path, start_time, should_loop,
           video_duration, output_path, video_info):
    """
    Builds and runs the FFmpeg command. Both video and audio filters
    live inside a single -filter_complex to avoid stream-mapping bugs.
    """
    vf = _video_filter(video_info["width"], video_info["height"])

    # Fade duration: 0.3s or 1/6 of duration for very short clips
    fade = min(0.3, video_duration / 6.0)
    fade_out_st = max(0.0, video_duration - fade)

    # Build a unified filter_complex with both video and audio chains
    audio_filters = (
        f"aresample=44100,"               # ensure consistent sample rate
        f"dynaudnorm=p=0.9:s=5,"          # normalize volume
        f"afade=t=in:d={fade:.3f},"        # fade in
        f"afade=t=out:st={fade_out_st:.3f}:d={fade:.3f},"
        f"asetpts=PTS-STARTPTS"
    )

    filter_complex = (
        f"[1:v]{vf},setpts=PTS-STARTPTS[vout];"
        f"[0:a]{audio_filters}[aout]"
    )

    cmd = ["ffmpeg", "-y"]

    # Audio input (with seeking or looping)
    if should_loop:
        cmd += ["-stream_loop", "-1"]
    elif start_time > 0.01:
        cmd += ["-ss", f"{start_time:.3f}"]

    cmd += ["-i", audio_path]          # input 0 = audio
    cmd += ["-i", video_path]          # input 1 = video

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-shortest",                   # stop when the shortest stream ends
        "-t", f"{video_duration:.3f}",
        output_path,
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{r.stderr[-2000:]}")

    # Verify the output actually has an audio stream with data
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,duration",
         "-of", "json", output_path],
        capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)
    if not info.get("streams"):
        raise RuntimeError("Output file has no audio stream — render failed silently.")

# ── Report logger ────────────────────────────────────────────────────────────
def append_report(video_name, music_name, start_time, looped, duration, output_name):
    """Appends one line per generated video to report.txt (never overwrites)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "loop" if looped else f"{start_time:.2f}s"
    line = f"[{ts}]  {output_name}  |  video={video_name}  music={music_name}  start={mode}  dur={duration:.2f}s"
    with report_lock:
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ── File scanner ─────────────────────────────────────────────────────────────
def scan_files(directory, exts):
    if not os.path.isdir(directory):
        return []
    return sorted(
        e.path for e in os.scandir(directory)
        if e.is_file() and os.path.splitext(e.name)[1].lower() in exts
    )

# ── Task Worker ──────────────────────────────────────────────────────────────
progress_counter = 0
progress_lock = threading.Lock()

def process_task(vpath, apath, peak_index, total_tasks):
    global progress_counter
    vname = os.path.basename(vpath)
    aname = os.path.basename(apath)
    
    try:
        info = get_video_info(vpath)
        dur = info["duration"]
        dur_str = format_duration(dur)
        sv = sanitize_name(vpath)
        sa = sanitize_name(apath)

        log_info(f"Analyzing audio '{aname}' (peak {peak_index+1}) for video '{vname}'...")
        t0, loop = analyze_audio(apath, dur, peak_index=peak_index)

        # Append suffix to denote the music version (peak index)
        sa_with_peak = f"{sa}_v{peak_index+1}"
        out = unique_output_path(sv, dur_str, sa_with_peak)
        outname = os.path.basename(out)
        
        log_info(f"Rendering: {vname} + {aname} (peak {peak_index+1}) -> {outname}")
        render(vpath, apath, t0, loop, dur, out, info)
        append_report(vname, f"{aname} (peak {peak_index+1})", t0, loop, dur, outname)
        log_success(f"Successfully generated: {outname}")
        
        with progress_lock:
            progress_counter += 1
            pct = int(progress_counter / total_tasks * 100)
            print(f"{_OK}[PROGRESS]{_RST} {progress_counter}/{total_tasks} ({pct}%) completed.")
        return True
    except Exception as e:
        log_error(f"Failed to process {vname} with {aname} (peak {peak_index+1}): {e}")
        with progress_lock:
            progress_counter += 1
        return False

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{_INF}╔══════════════════════════════════════════════╗{_RST}")
    print(f"{_INF}║   TikTok Video Autogenerator (tkvideobulk)   ║{_RST}")
    print(f"{_INF}╚══════════════════════════════════════════════╝{_RST}\n")

    validate_ffmpeg()
    ensure_dirs()

    videos = scan_files(IN_VIDEO_DIR, SUPPORTED_VIDEO_EXTS)
    audios = scan_files(IN_WAV_DIR, SUPPORTED_AUDIO_EXTS)

    if not videos:
        log_warning(f"No videos in '{IN_VIDEO_DIR}/' — add .mp4/.mov/.mkv/.avi files.")
    if not audios:
        log_warning(f"No audio in '{IN_WAV_DIR}/' — add .wav/.mp3/.flac/.m4a files.")
    if not videos or not audios:
        return

    # Pair each video with every music track, and generate 2 videos per combination (using the 2 highest energy peaks)
    tasks = []
    for vpath in videos:
        for apath in audios:
            tasks.append((vpath, apath, 0)) # 1st peak
            tasks.append((vpath, apath, 1)) # 2nd peak

    total_tasks = len(tasks)
    log_info(f"Found {len(videos)} video(s) and {len(audios)} audio(s). Generating 2 clips each per combination.")
    log_info(f"Total outputs to render: {total_tasks}\n")

    # Use CPU count for Apple Silicon multi-threading
    cpu_count = os.cpu_count() or 4
    # Max out at 4 or cpu_count to avoid overwhelming disk I/O, but allow scaling
    max_workers = max(1, cpu_count)
    log_info(f"Starting parallel execution using ThreadPoolExecutor with {max_workers} workers.\n")

    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_task, vpath, apath, peak_index, total_tasks) for vpath, apath, peak_index in tasks]
        for fut in futures:
            if fut.result():
                ok += 1
            else:
                fail += 1

    print(f"\n\n{_INF}══════════════════════════════════════════════{_RST}")
    print(f"  {_OK}Done!{_RST}  Generated: {ok}  |  Failed: {fail}")
    print(f"  Output folder: {OUT_VIDEO_DIR}/")
    print(f"{_INF}══════════════════════════════════════════════{_RST}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
