"""Dependency-light video helpers shared by legacy and refactored solver paths."""

from __future__ import annotations

import base64
import html
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.infra.artifacts import _ensure_parent
from src.infra.solver_config import _cfg_get
from src.infra.utils import _safe_float


def _looks_like_video_url(url: str) -> bool:
    raw = html.unescape((url or "").strip())
    if not raw:
        return False
    lowered = raw.lower()
    if lowered.startswith("blob:"):
        return False
    parsed = urlparse(lowered)
    path = parsed.path or ""
    if re.search(r"\.(mp4|webm|mov|m4v|m3u8)$", path, flags=re.I):
        return True
    if re.search(r"\.(woff2?|ttf|otf|css|js|map|png|jpe?g|gif|svg|ico)$", path, flags=re.I):
        return False
    return ("video" in path) or ("video" in parsed.query)


def _is_probably_mp4(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(64)
    except Exception:
        return False
    if len(head) < 12:
        return False
    return b"ftyp" in head[:16]


def _is_video_decodable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import cv2  # type: ignore
    except Exception:
        # Without cv2 we cannot verify decode-ability; assume False (fail-closed).
        return False

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        probe_positions = [0]
        if frame_count > 2:
            probe_positions.append(max(0, frame_count // 2))
            probe_positions.append(max(0, frame_count - 2))
        for pos in probe_positions:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(pos))
            except Exception:
                pass
            ok, _ = cap.read()
            if not ok:
                return False
        return True
    finally:
        cap.release()


def _probe_video_stream_meta(path: Path) -> Tuple[int, int, float, int]:
    try:
        import cv2  # type: ignore
    except Exception:
        return 0, 0, 0.0, 0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0, 0.0, 0
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return width, height, fps, frames
    finally:
        cap.release()


def _quality_preserving_scale_candidates(
    scales: List[float],
    src_w: int,
    src_h: int,
    min_width: int,
    min_short_side: int,
) -> List[float]:
    if src_w <= 0 or src_h <= 0:
        return scales

    min_width = max(2, int(min_width))
    min_short_side = max(2, int(min_short_side))
    short_side = min(src_w, src_h)

    width_floor = min_width / float(src_w)
    short_floor = min_short_side / float(short_side) if short_side > 0 else 0.0
    scale_floor = max(0.1, min(1.0, max(width_floor, short_floor)))

    filtered: List[float] = []
    for raw in scales:
        s = max(0.1, min(1.0, float(raw)))
        if s + 1e-6 >= scale_floor:
            filtered.append(s)

    if not filtered:
        filtered = [scale_floor]
    elif all(abs(s - scale_floor) > 1e-3 for s in filtered):
        filtered.append(scale_floor)

    uniq = sorted({round(s, 4) for s in filtered}, reverse=True)
    return [float(s) for s in uniq]


def _extract_reference_frame_inline_parts(
    video_file: Path,
    cfg: Dict[str, Any],
    trigger_video_mb: float,
) -> Tuple[List[Dict[str, Any]], int]:
    enabled = bool(_cfg_get(cfg, "gemini.reference_frames_enabled", True))
    if not enabled or video_file is None or not video_file.exists():
        return [], 0

    always = bool(_cfg_get(cfg, "gemini.reference_frames_always", False))
    trigger_mb = max(0.1, float(_cfg_get(cfg, "gemini.reference_frame_attach_when_video_mb_le", 2.5)))
    if not always and trigger_video_mb > trigger_mb:
        return [], 0

    try:
        import cv2  # type: ignore
    except Exception:
        return [], 0

    frame_count = max(1, int(_cfg_get(cfg, "gemini.reference_frame_count", 2)))
    max_side = max(240, int(_cfg_get(cfg, "gemini.reference_frame_max_side", 960)))
    jpeg_quality = max(50, min(95, int(_cfg_get(cfg, "gemini.reference_frame_jpeg_quality", 82))))
    max_total_bytes = max(64 * 1024, int(float(_cfg_get(cfg, "gemini.reference_frame_max_total_kb", 420)) * 1024))

    raw_positions = _cfg_get(cfg, "gemini.reference_frame_positions", [0.2, 0.55, 0.85])
    pos_list: List[float] = []
    if isinstance(raw_positions, list):
        for raw in raw_positions:
            try:
                v = float(raw)
            except Exception:
                continue
            if 0.0 <= v <= 1.0:
                pos_list.append(v)
    if not pos_list:
        step = 1.0 / float(frame_count + 1)
        pos_list = [step * (i + 1) for i in range(frame_count)]

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return [], 0

    try:
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frames_total <= 0:
            return [], 0

        indices: List[int] = []
        for p in pos_list:
            idx = int(round((frames_total - 1) * max(0.0, min(1.0, p))))
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= frame_count:
                break
        if not indices:
            return [], 0

        parts: List[Dict[str, Any]] = []
        total_bytes = 0
        for idx in indices:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            except Exception:
                pass
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                continue
            largest = max(h, w)
            if largest > max_side:
                scale = max_side / float(largest)
                nw = max(2, int(round(w * scale)))
                nh = max(2, int(round(h * scale)))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            ok_enc, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok_enc:
                continue
            data = bytes(enc.tobytes())
            if not data:
                continue
            if total_bytes + len(data) > max_total_bytes:
                break
            total_bytes += len(data)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                }
            )
        return parts, total_bytes
    finally:
        cap.release()


def _ensure_even(value: int, minimum: int = 2) -> int:
    v = max(int(minimum), int(value))
    return v if v % 2 == 0 else v + 1


def _parse_float_list(value: Any, fallback: List[float]) -> List[float]:
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            try:
                n = float(item)
                if n > 0:
                    out.append(n)
            except Exception:
                continue
        if out:
            return out
        return list(fallback)
    if isinstance(value, str):
        out = []
        for raw in value.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                n = float(raw)
                if n > 0:
                    out.append(n)
            except Exception:
                continue
        if out:
            return out
    return list(fallback)


def _opencv_available() -> bool:
    try:
        import cv2  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_ffmpeg_binary() -> Optional[str]:
    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    user_profile = str(os.environ.get("USERPROFILE", "") or "").strip()
    # Build the candidate list with platform-specific absolute paths first,
    # so they take priority over bare names resolved via shutil.which().
    # On Windows (LOCALAPPDATA set), WinGet/scoop paths come first.
    # On Linux, /usr/bin/ffmpeg and /usr/local/bin/ffmpeg come first.
    # Bare names like "ffmpeg" are tried last as a PATH fallback.
    candidates: list[str] = []
    if local_app_data:
        candidates.extend(
            [
                str(Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"),
                str(Path(local_app_data) / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" / "ffmpeg-8.0-full_build" / "bin" / "ffmpeg.exe"),
            ]
        )
    if user_profile:
        candidates.append(str(Path(user_profile) / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffmpeg.exe"))
    if not local_app_data and not user_profile:
        candidates.extend(["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"])
    candidates.extend(
        [
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]
    )
    candidates.extend(["ffmpeg", "ffmpeg.exe"])
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        try:
            p = Path(candidate)
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def _resolve_ffprobe_binary(ffmpeg_bin: Optional[str] = None) -> Optional[str]:
    if ffmpeg_bin:
        try:
            ffmpeg_path = Path(ffmpeg_bin)
            probe_name = "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
            sibling = ffmpeg_path.with_name(probe_name)
            if sibling.exists() and sibling.is_file():
                return str(sibling)
        except Exception:
            pass
    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    user_profile = str(os.environ.get("USERPROFILE", "") or "").strip()
    candidates = [
        "ffprobe",
        "ffprobe.exe",
        "/usr/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "C:\\ffmpeg\\bin\\ffprobe.exe",
    ]
    if local_app_data:
        candidates.extend(
            [
                str(Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffprobe.exe"),
                str(Path(local_app_data) / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" / "ffmpeg-8.0-full_build" / "bin" / "ffprobe.exe"),
            ]
        )
    if user_profile:
        candidates.append(str(Path(user_profile) / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffprobe.exe"))
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        try:
            p = Path(candidate)
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def _probe_video_duration_seconds(video_file: Path, ffmpeg_bin: Optional[str] = None) -> float:
    if video_file is None or not video_file.exists():
        return 0.0

    try:
        _, _, fps, frames = _probe_video_stream_meta(video_file)
        if fps > 0 and frames > 0:
            duration = float(frames) / float(fps)
            if duration > 0.2:
                return duration
    except Exception:
        pass

    ffprobe_bin = _resolve_ffprobe_binary(ffmpeg_bin=ffmpeg_bin)
    if not ffprobe_bin:
        return 0.0
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_file),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return 0.0
        out = (proc.stdout or "").strip().splitlines()
        if not out:
            return 0.0
        duration = float(out[-1].strip())
        if duration > 0.2:
            return duration
    except Exception:
        return 0.0
    return 0.0


def _split_video_for_upload(video_file: Path, cfg: Dict[str, Any]) -> List[Path]:
    if video_file is None or not video_file.exists():
        return []
    if not bool(_cfg_get(cfg, "gemini.split_upload_enabled", True)):
        return []

    size_bytes = int(video_file.stat().st_size)
    size_mb = size_bytes / (1024 * 1024)
    trigger_mb = max(1.0, float(_cfg_get(cfg, "gemini.split_upload_only_if_larger_mb", 14.0)))
    if size_mb <= trigger_mb:
        return []

    chunk_max_mb = max(2.0, float(_cfg_get(cfg, "gemini.split_upload_chunk_max_mb", 6.0)))
    max_chunks = max(2, int(_cfg_get(cfg, "gemini.split_upload_max_chunks", 4)))
    split_count = int(math.ceil(size_mb / chunk_max_mb))
    split_count = max(2, min(max_chunks, split_count))
    if split_count <= 1:
        return []

    ffmpeg_bin = _resolve_ffmpeg_binary()
    if not ffmpeg_bin:
        print("[video] split upload skipped: ffmpeg not available.")
        return []

    duration_sec = _probe_video_duration_seconds(video_file, ffmpeg_bin=ffmpeg_bin)
    if duration_sec <= 0.2:
        print("[video] split upload skipped: could not determine video duration.")
        return []

    stem = video_file.stem
    parent = video_file.parent
    out_files = [parent / f"{stem}_upload_part{i + 1:02d}.mp4" for i in range(split_count)]
    if all(p.exists() and p.stat().st_size > 0 and _is_probably_mp4(p) for p in out_files):
        total_mb = sum(float(p.stat().st_size) for p in out_files) / (1024 * 1024)
        print(
            f"[video] using cached split upload parts: {len(out_files)} parts "
            f"({total_mb:.1f} MB total)."
        )
        return out_files

    for stale in parent.glob(f"{stem}_upload_part*.mp4"):
        try:
            stale.unlink(missing_ok=True)
        except Exception:
            pass

    chunk_duration = duration_sec / float(split_count)
    use_reencode_on_copy_fail = bool(_cfg_get(cfg, "gemini.split_upload_reencode_on_copy_fail", True))
    print(
        f"[video] splitting upload video into {split_count} parts "
        f"(source={size_mb:.1f} MB, duration={duration_sec:.1f}s)."
    )
    produced: List[Path] = []
    for idx, out_path in enumerate(out_files):
        start_sec = idx * chunk_duration
        if idx == split_count - 1:
            dur_sec = max(0.2, duration_sec - start_sec)
        else:
            dur_sec = max(0.2, chunk_duration)

        cmd_copy = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{dur_sec:.3f}",
            "-i",
            str(video_file),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        ok = False
        try:
            proc = subprocess.run(
                cmd_copy,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                timeout=240,
            )
            ok = (
                proc.returncode == 0
                and out_path.exists()
                and out_path.stat().st_size > 0
                and _is_probably_mp4(out_path)
            )
        except Exception:
            ok = False

        if not ok and use_reencode_on_copy_fail:
            cmd_enc = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_sec:.3f}",
                "-t",
                f"{dur_sec:.3f}",
                "-i",
                str(video_file),
                "-an",
                "-sn",
                "-dn",
                "-c:v",
                "libx264",
                "-preset",
                "faster",
                "-crf",
                "21",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            try:
                proc = subprocess.run(
                    cmd_enc,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True,
                    timeout=240,
                )
                ok = (
                    proc.returncode == 0
                    and out_path.exists()
                    and out_path.stat().st_size > 0
                    and _is_probably_mp4(out_path)
                )
            except Exception:
                ok = False

        if not ok:
            print(f"[video] split chunk failed at part {idx + 1}; falling back to single-file flow.")
            for p in out_files:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            return []

        produced.append(out_path)
        try:
            part_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[video] split part {idx + 1}/{split_count}: {out_path.name} ({part_mb:.1f} MB)")
        except Exception:
            pass

    if len(produced) != split_count:
        return []
    return produced


def _segment_chunks(
    segments: List[Dict[str, Any]],
    max_per_chunk: int,
    *,
    max_window_sec: float = 0.0,
) -> List[List[Dict[str, Any]]]:
    if not segments:
        return []
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_start: Optional[float] = None
    window_limit = max(0.0, float(max_window_sec or 0.0))

    for seg in segments:
        seg_start = _safe_float(seg.get("start_sec"), 0.0)
        seg_end = max(seg_start, _safe_float(seg.get("end_sec"), seg_start))
        if not current:
            current = [seg]
            current_start = seg_start
            continue

        exceeds_count = len(current) >= max(1, int(max_per_chunk))
        exceeds_window = False
        if window_limit > 0.0 and current_start is not None:
            exceeds_window = (seg_end - current_start) > window_limit

        if exceeds_count or exceeds_window:
            chunks.append(current)
            current = [seg]
            current_start = seg_start
            continue

        current.append(seg)

    if current:
        chunks.append(current)
    return chunks


def _extract_video_window(
    src_video: Path,
    out_video: Path,
    start_sec: float,
    end_sec: float,
    ffmpeg_bin: Optional[str] = None,
) -> bool:
    ffmpeg_path = ffmpeg_bin or _resolve_ffmpeg_binary()
    if not ffmpeg_path:
        return False
    if src_video is None or not src_video.exists():
        return False
    if end_sec <= start_sec:
        return False

    duration = max(0.2, float(end_sec - start_sec))
    try:
        _ensure_parent(out_video)
        if out_video.exists():
            out_video.unlink()
    except Exception:
        pass

    copy_cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src_video),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    try:
        proc = subprocess.run(
            copy_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=240,
        )
        if proc.returncode == 0 and out_video.exists() and out_video.stat().st_size > 0 and _is_probably_mp4(out_video):
            return True
    except Exception:
        pass

    encode_cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src_video),
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    try:
        proc = subprocess.run(
            encode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=300,
        )
        return bool(
            proc.returncode == 0
            and out_video.exists()
            and out_video.stat().st_size > 0
            and _is_probably_mp4(out_video)
        )
    except Exception:
        return False


def _transcode_video_ffmpeg(
    src: Path,
    dst: Path,
    scale: float,
    target_fps: float,
    min_width: int,
    ffmpeg_bin: Optional[str] = None,
) -> Tuple[bool, str]:
    ffmpeg_path = ffmpeg_bin or _resolve_ffmpeg_binary()
    if not ffmpeg_path:
        return False, "ffmpeg binary not found"

    vf = (
        f"scale=max({min_width}\\,trunc(iw*{float(scale):.4f}/2)*2):-2,"
        f"fps={max(1.0, float(target_fps)):.2f}"
    )
    codec_attempts: List[List[str]] = [
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"],
        ["-c:v", "mpeg4", "-q:v", "10"],
    ]
    last_err = ""
    for codec_opts in codec_attempts:
        try:
            _ensure_parent(dst)
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        cmd = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            vf,
            "-an",
            "-sn",
            "-dn",
            *codec_opts,
            "-movflags",
            "+faststart",
            str(dst),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                timeout=420,
            )
        except Exception as exc:
            last_err = str(exc)
            continue
        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0 and _is_probably_mp4(dst):
            return True, ""
        stderr_snippet = (proc.stderr or "").strip()
        if stderr_snippet:
            stderr_snippet = stderr_snippet.splitlines()[-1]
        last_err = stderr_snippet or f"ffmpeg exit code {proc.returncode}"
    return False, last_err


def _transcode_video_cv2(
    src: Path,
    dst: Path,
    scale: float,
    target_fps: float,
    min_width: int,
) -> bool:
    try:
        import cv2  # type: ignore
    except Exception:
        return False

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_w <= 0 or src_h <= 0:
            return False
        if src_fps <= 0.1 or src_fps > 240:
            src_fps = 24.0

        scaled_w = max(min_width, int(round(src_w * float(scale))))
        scaled_w = _ensure_even(scaled_w, minimum=min_width)
        scaled_h = int(round(src_h * (scaled_w / float(src_w))))
        scaled_h = _ensure_even(scaled_h, minimum=2)

        target_fps = max(1.0, min(float(target_fps), src_fps))
        frame_interval = max(1, int(round(src_fps / target_fps)))
        out_fps = max(1.0, src_fps / frame_interval)

        _ensure_parent(dst)
        if dst.exists():
            dst.unlink()

        writer = None
        for codec in ("mp4v", "avc1", "H264", "XVID"):
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                candidate = cv2.VideoWriter(str(dst), fourcc, out_fps, (scaled_w, scaled_h))
                if candidate.isOpened():
                    writer = candidate
                    break
                candidate.release()
            except Exception:
                continue
        if writer is None:
            return False

        frame_idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_interval > 1 and (frame_idx % frame_interval) != 0:
                frame_idx += 1
                continue
            if frame.shape[1] != scaled_w or frame.shape[0] != scaled_h:
                frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
            frame_idx += 1
        writer.release()
        if written <= 0:
            return False
    finally:
        cap.release()

    return dst.exists() and dst.stat().st_size > 0 and _is_probably_mp4(dst)


def _maybe_optimize_video_for_upload(video_file: Path, cfg: Dict[str, Any]) -> Path:
    if video_file is None or not video_file.exists():
        return video_file

    enabled = bool(_cfg_get(cfg, "gemini.optimize_video_for_upload", True))
    if not enabled:
        return video_file

    size_bytes = int(video_file.stat().st_size)
    size_mb = size_bytes / (1024 * 1024)
    trigger_mb = max(1.0, float(_cfg_get(cfg, "gemini.optimize_video_only_if_larger_mb", 8.0)))
    if size_mb <= trigger_mb:
        return video_file

    target_mb = max(1.0, float(_cfg_get(cfg, "gemini.optimize_video_target_mb", 15.0)))
    target_bytes = int(target_mb * 1024 * 1024)
    target_fps = max(1.0, float(_cfg_get(cfg, "gemini.optimize_video_target_fps", 10.0)))
    min_fps = max(1.0, float(_cfg_get(cfg, "gemini.optimize_video_min_fps", 8.0)))
    target_fps = max(min_fps, target_fps)
    min_width = max(160, int(_cfg_get(cfg, "gemini.optimize_video_min_width", 320)))
    min_short_side = max(160, int(_cfg_get(cfg, "gemini.optimize_video_min_short_side", 320)))
    prefer_ffmpeg = bool(_cfg_get(cfg, "gemini.optimize_video_prefer_ffmpeg", True))
    scales = _parse_float_list(
        _cfg_get(cfg, "gemini.optimize_video_scale_candidates", [0.75, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2]),
        [0.75, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2],
    )
    src_w, src_h, src_fps, _ = _probe_video_stream_meta(video_file)
    scales = _quality_preserving_scale_candidates(
        scales=scales,
        src_w=src_w,
        src_h=src_h,
        min_width=min_width,
        min_short_side=min_short_side,
    )

    out_file = video_file.with_name(f"{video_file.stem}_upload_opt.mp4")
    if out_file.exists():
        try:
            out_size = int(out_file.stat().st_size)
            if out_size > 0 and _is_probably_mp4(out_file) and out_size <= target_bytes:
                print(
                    f"[video] using cached optimized upload file: {out_file} "
                    f"({out_size / (1024 * 1024):.1f} MB)"
                )
                return out_file
        except Exception:
            pass

    src_meta_note = ""
    if src_w > 0 and src_h > 0:
        fps_note = f", {src_fps:.1f}fps" if src_fps > 0 else ""
        src_meta_note = f", source={src_w}x{src_h}{fps_note}"
    print(
        f"[video] optimizing video for upload: {video_file.name} "
        f"({size_mb:.1f} MB -> target <= {target_mb:.1f} MB{src_meta_note})"
    )
    cv2_available = _opencv_available()
    ffmpeg_bin = _resolve_ffmpeg_binary()
    if not cv2_available and ffmpeg_bin:
        print(f"[video] OpenCV unavailable; using ffmpeg optimizer backend: {ffmpeg_bin}")
    elif not cv2_available and not ffmpeg_bin:
        print("[video] OpenCV and ffmpeg are unavailable; cannot optimize upload video.")
    elif prefer_ffmpeg and ffmpeg_bin and cv2_available:
        print(f"[video] preferring ffmpeg optimizer backend: {ffmpeg_bin}")
    candidates: List[Path] = []
    best_path: Optional[Path] = None
    best_size = size_bytes
    ffmpeg_last_error = ""

    for scale in scales:
        scale = max(0.1, min(1.0, float(scale)))
        suffix = int(round(scale * 100))
        cand = video_file.with_name(f"{video_file.stem}_upload_opt_s{suffix}.mp4")
        ok = False
        backend_used: Optional[str] = None
        backend_order: List[str] = []
        if prefer_ffmpeg and ffmpeg_bin:
            backend_order.append("ffmpeg")
        if cv2_available:
            backend_order.append("cv2")
        if ffmpeg_bin and "ffmpeg" not in backend_order:
            backend_order.append("ffmpeg")

        for backend_name in backend_order:
            if backend_name == "cv2":
                try:
                    ok = _transcode_video_cv2(
                        src=video_file,
                        dst=cand,
                        scale=scale,
                        target_fps=target_fps,
                        min_width=min_width,
                    )
                except Exception:
                    ok = False
                if ok:
                    backend_used = "cv2"
                    break
                continue

            try:
                ok, ffmpeg_err = _transcode_video_ffmpeg(
                    src=video_file,
                    dst=cand,
                    scale=scale,
                    target_fps=target_fps,
                    min_width=min_width,
                    ffmpeg_bin=ffmpeg_bin,
                )
                if ok:
                    backend_used = "ffmpeg"
                    break
                if ffmpeg_err:
                    ffmpeg_last_error = ffmpeg_err
            except Exception as exc:
                ffmpeg_last_error = str(exc)
                ok = False
        if not ok:
            continue
        candidates.append(cand)
        try:
            cand_size = int(cand.stat().st_size)
        except Exception:
            continue
        backend_note = f" ({backend_used})" if backend_used else ""
        print(f"[video] optimized candidate scale={scale:.2f}: {cand_size / (1024 * 1024):.1f} MB{backend_note}")
        if cand_size < best_size:
            best_size = cand_size
            best_path = cand
        if cand_size <= target_bytes:
            break

    if best_path is None:
        if ffmpeg_last_error:
            print(f"[video] ffmpeg optimizer failed: {ffmpeg_last_error}")
        print("[video] upload optimization not available; using original video.")
        return video_file

    try:
        if out_file.exists():
            out_file.unlink()
        best_path.replace(out_file)
    except Exception:
        out_file = best_path

    for cand in candidates:
        if cand == out_file:
            continue
        try:
            cand.unlink(missing_ok=True)
        except Exception:
            continue

    out_size = int(out_file.stat().st_size) if out_file.exists() else size_bytes
    if out_size >= size_bytes:
        print("[video] optimization did not reduce size enough; using original video.")
        return video_file
    if out_size > target_bytes:
        print(
            f"[video] optimized upload video remains above target: "
            f"{out_size / (1024 * 1024):.1f} MB > {target_mb:.1f} MB "
            "(quality-preserving floor/backends prevented further reduction)."
        )
    print(
        f"[video] optimized upload video ready: {out_file} "
        f"({out_size / (1024 * 1024):.1f} MB)"
    )
    return out_file


def _prepare_repair_video_clip(
    video_file: Path,
    segments: List[Dict[str, Any]],
    target_indices: List[int],
    out_dir: Path,
    episode_id: str = "",
    *,
    pad_sec: float = 2.0,
    cdp_max_mb: float = 45.0,
    ffmpeg_bin: Optional[str] = None,
) -> Optional[Path]:
    """Create a time-windowed video clip scoped to the overlong segments for repair.

    Returns the path to a small clip that covers the overlong segments (plus
    padding), suitable for uploading via CDP where the 50 MB limit applies.
    Falls back to the ``_upload_opt`` file when clipping is not possible.
    Returns ``None`` only when no usable file can be produced.
    """
    if video_file is None or not video_file.exists():
        return None
    if not target_indices or not segments:
        return None

    target_set = set(int(idx) for idx in target_indices)
    matched = [
        seg for seg in segments
        if int(seg.get("segment_index", 0) or 0) in target_set
    ]
    if not matched:
        return None

    window_start = max(
        0.0,
        min(_safe_float(seg.get("start_sec"), 0.0) for seg in matched) - pad_sec,
    )
    window_end = (
        max(_safe_float(seg.get("end_sec"), 0.0) for seg in matched) + pad_sec
    )
    if window_end <= window_start:
        window_end = window_start + 1.0

    # Prefer the optimized upload video as the source (smaller, faster clip).
    opt_path = video_file.with_name(video_file.stem + "_upload_opt.mp4")
    src = opt_path if (opt_path.exists() and opt_path.stat().st_size > 0) else video_file

    token = str(episode_id or "unknown").strip()[-24:] or "unknown"
    clip_name = f"video_{token}_repair_clip.mp4"
    clip_path = out_dir / clip_name

    clipped = _extract_video_window(
        src_video=src,
        out_video=clip_path,
        start_sec=window_start,
        end_sec=window_end,
        ffmpeg_bin=ffmpeg_bin,
    )
    if clipped and clip_path.exists() and clip_path.stat().st_size > 0:
        clip_mb = clip_path.stat().st_size / (1024 * 1024)
        if clip_mb <= cdp_max_mb:
            print(
                f"[video] repair clip ready: {clip_path.name} "
                f"({clip_mb:.1f} MB, window={window_start:.1f}-{window_end:.1f}s, "
                f"targets={sorted(target_set)})"
            )
            return clip_path
        # Clip too large -- clean up and fall through to _upload_opt.
        try:
            clip_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Fall back to _upload_opt if it exists and is small enough.
    if opt_path.exists():
        opt_mb = opt_path.stat().st_size / (1024 * 1024)
        if opt_mb <= cdp_max_mb:
            print(
                f"[video] repair clip fallback to optimized upload: "
                f"{opt_path.name} ({opt_mb:.1f} MB)"
            )
            return opt_path

    # Last resort: original video only if small enough.
    try:
        orig_mb = video_file.stat().st_size / (1024 * 1024)
        if orig_mb <= cdp_max_mb:
            return video_file
    except Exception:
        pass

    print(
        "[video] repair clip: no file small enough for CDP transfer "
        f"(limit={cdp_max_mb:.0f} MB)"
    )
    return None


__all__ = [
    "_looks_like_video_url",
    "_is_probably_mp4",
    "_is_video_decodable",
    "_probe_video_stream_meta",
    "_quality_preserving_scale_candidates",
    "_extract_reference_frame_inline_parts",
    "_ensure_even",
    "_parse_float_list",
    "_opencv_available",
    "_resolve_ffmpeg_binary",
    "_resolve_ffprobe_binary",
    "_probe_video_duration_seconds",
    "_split_video_for_upload",
    "_segment_chunks",
    "_extract_video_window",
    "_transcode_video_ffmpeg",
    "_transcode_video_cv2",
    "_maybe_optimize_video_for_upload",
    "_prepare_repair_video_clip",
]
