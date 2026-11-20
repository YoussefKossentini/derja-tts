"""Cut long recordings of one speaker into training clips with draft transcripts.

    python scripts/segment_audio.py recordings/ --out data/youssef --speaker youssef

What it does, per input file:
  1. finds speech with Silero VAD
  2. groups it into 4-12 s clips, cutting at pauses
  3. writes each clip as mono WAV (24 kHz by default) into <out>/wavs/
  4. drafts a transcript with a Tunisian ASR model (skip with --no-asr)
  5. writes <out>/metadata.csv as  id|raw transcript|normalized transcript

The ASR draft WILL have mistakes. Open metadata.csv, fix column 2 by hand while
listening, then refresh column 3 with:
    derja-tts metadata <out>/metadata.csv

Inputs: .wav / .flac / .mp3 / .m4a / .ogg, or folders of them. For videos, pull
the audio first:  ffmpeg -i video.mp4 -vn -ac 1 -ar 48000 narration.wav
Background music ruins TTS data: only use narration recorded without music.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from derja_tts.audio import group_speech, pad, safe_id, write_metadata  # noqa: E402
from derja_tts.text import Normalizer  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
VAD_SR = 16000


def find_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        elif p.suffix.lower() in AUDIO_EXT:
            files.append(p)
    return files


def speech_regions(wav16: np.ndarray, vad_model, min_silence_ms: int) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps

    stamps = get_speech_timestamps(
        torch.from_numpy(wav16), vad_model, sampling_rate=VAD_SR,
        min_silence_duration_ms=min_silence_ms, return_seconds=True,
    )
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def load_asr(model_id: str, device: str | None):
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=model_id, device=device)


def transcribe(asr, clip16: np.ndarray, language: str | None) -> str:
    kwargs = {"generate_kwargs": {"language": language, "task": "transcribe"}} if language else {}
    try:
        out = asr({"raw": clip16, "sampling_rate": VAD_SR}, **kwargs)
    except (TypeError, ValueError):
        out = asr({"raw": clip16, "sampling_rate": VAD_SR})  # non-Whisper models
    return out["text"].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="audio files or folders")
    ap.add_argument("--out", required=True, help="output folder (gets wavs/ and metadata.csv)")
    ap.add_argument("--speaker", required=True, help="speaker id, used as clip-id prefix")
    ap.add_argument("--sr", type=int, default=24000, help="output sample rate (default 24000)")
    ap.add_argument("--min", dest="min_dur", type=float, default=4.0)
    ap.add_argument("--max", dest="max_dur", type=float, default=12.0)
    ap.add_argument("--max-gap", type=float, default=0.8, help="longest pause kept inside a clip (s)")
    ap.add_argument("--min-silence-ms", type=int, default=300, help="VAD: silence that splits speech")
    ap.add_argument("--pad", type=float, default=0.15, help="silence kept around each clip (s)")
    ap.add_argument("--asr-model", default="FARUKxAUTO/FarukSTT",
                    help="HF model id of a Tunisian ASR model (Whisper-style pipeline)")
    ap.add_argument("--asr-language", default="arabic", help="Whisper language hint, '' to disable")
    ap.add_argument("--no-asr", action="store_true", help="write empty transcripts instead")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (default: auto)")
    ap.add_argument("--append", action="store_true", help="add to an existing metadata.csv")
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    from silero_vad import load_silero_vad
    from tqdm import tqdm

    files = find_inputs(args.inputs)
    if not files:
        sys.exit("No audio files found.")

    out_dir = Path(args.out)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"

    rows: list[tuple[str, str, str]] = []
    if args.append and meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            parts = (line.split("|") + ["", ""])[:3]
            rows.append((parts[0], parts[1], parts[2]))

    vad = load_silero_vad()
    asr = None if args.no_asr else load_asr(args.asr_model, args.device)
    norm = Normalizer()

    kept_seconds = 0.0
    for f in files:
        wav, _ = librosa.load(f, sr=args.sr, mono=True)
        wav16 = librosa.resample(wav, orig_sr=args.sr, target_sr=VAD_SR)
        total = len(wav) / args.sr

        regions = speech_regions(wav16, vad, args.min_silence_ms)
        clips = group_speech(regions, args.min_dur, args.max_dur, args.max_gap)
        stem = safe_id(f"{args.speaker}_{f.stem}")
        print(f"{f.name}: {total / 60:.1f} min, {len(regions)} speech regions -> {len(clips)} clips")

        for k, region in enumerate(tqdm(clips, leave=False)):
            s, e = pad(region, args.pad, total)
            clip = wav[int(s * args.sr):int(e * args.sr)]
            clip_id = f"{stem}_{k:04d}"
            sf.write(wav_dir / f"{clip_id}.wav", clip, args.sr, subtype="PCM_16")
            raw = ""
            if asr is not None:
                clip16 = wav16[int(s * VAD_SR):int(e * VAD_SR)]
                raw = transcribe(asr, clip16, args.asr_language or None)
            rows.append((clip_id, raw, norm(raw) if raw else ""))
            kept_seconds += e - s

    write_metadata(meta_path, rows)
    print(f"\n{len(rows)} clips, {kept_seconds / 60:.1f} min kept -> {meta_path}")
    print("Next: fix column 2 by ear, then run  derja-tts metadata", meta_path)


if __name__ == "__main__":
    main()
