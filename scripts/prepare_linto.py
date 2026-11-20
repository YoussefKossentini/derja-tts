"""Turn the LinTO Tunisian audio dataset (CC BY 4.0) into TTS training clips.

    python scripts/prepare_linto.py --out data/linto
    python scripts/prepare_linto.py --out data/linto --configs AmenyKH TunSwitchCS --max-hours 5

Writes:
  <out>/wavs/*.wav        mono, 24 kHz by default
  <out>/metadata.csv      id|raw transcript|normalized transcript
  <out>/attribution.csv   id,source_audio_id,config,license   (for credit + speaker-aware splits)

Dataset: https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn
Licence: CC BY 4.0. Credit LinTO / LINAGORA in anything you release (see README).
Each row holds an audio file and a list of segments (start, end, transcript).
We cut every segment that lasts between --min and --max seconds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from derja_tts.audio import safe_id, write_metadata  # noqa: E402
from derja_tts.text import Normalizer  # noqa: E402

DATASET_ID = "linagora/linto-dataset-audio-ar-tn"


def decode_audio(audio) -> tuple[np.ndarray, int]:
    """Works with both datasets<4 (dict) and datasets>=4 (torchcodec AudioDecoder)."""
    if isinstance(audio, dict) and "array" in audio:
        return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
    samples = audio.get_all_samples()
    data = samples.data.numpy()
    if data.ndim == 2:
        data = data.mean(axis=0)
    return data.astype(np.float32), int(samples.sample_rate)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=[None],
                    help="dataset configs (default: all). e.g. AmenyKH TunSwitchCS OneStory")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--min", dest="min_dur", type=float, default=4.0)
    ap.add_argument("--max", dest="max_dur", type=float, default=12.0)
    ap.add_argument("--max-hours", type=float, default=None, help="stop after this much audio")
    ap.add_argument("--streaming", action="store_true", help="stream instead of downloading ~2 GB")
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    from datasets import load_dataset
    from tqdm import tqdm

    out = Path(args.out)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    norm = Normalizer()

    rows: list[tuple[str, str, str]] = []
    attribution: list[tuple[str, str, str, str]] = []
    kept = 0.0
    budget = args.max_hours * 3600 if args.max_hours else float("inf")

    for config in args.configs:
        name = config or "default"
        ds = load_dataset(DATASET_ID, config, split=args.split, streaming=args.streaming)
        for ex in tqdm(ds, desc=name):
            if kept >= budget:
                break
            wav, sr = decode_audio(ex["audio"])
            if sr != args.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=args.sr)
            segments = ex.get("segments") or [
                {"start": 0.0, "end": len(wav) / args.sr, "transcript": ex.get("transcript", "")}
            ]
            base = safe_id(f"linto_{name}_{ex.get('audio_id', len(rows))}")
            for j, seg in enumerate(segments):
                start, end = float(seg["start"]), float(seg["end"])
                text = (seg.get("transcript") or "").strip()
                if not text or not (args.min_dur <= end - start <= args.max_dur):
                    continue
                clip = wav[int(start * args.sr):int(end * args.sr)]
                clip_id = f"{base}_{j:04d}"
                sf.write(wav_dir / f"{clip_id}.wav", clip, args.sr, subtype="PCM_16")
                rows.append((clip_id, text, norm(text)))
                attribution.append((clip_id, str(ex.get("audio_id", "")), name, "CC-BY-4.0"))
                kept += end - start

    write_metadata(out / "metadata.csv", rows)
    with open(out / "attribution.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source_audio_id", "config", "license"])
        w.writerows(attribution)
    print(f"{len(rows)} clips, {kept / 3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()
