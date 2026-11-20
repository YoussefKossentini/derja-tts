"""Command line entry point: `derja-tts <command>`.

    derja-tts normalize "3aslema, lyoum bech nfasserlkom el overfitting"
    derja-tts normalize --file script.txt
    derja-tts unknown --file script.txt          # Latin words to review
    derja-tts metadata data/my_voice/metadata.csv  # refresh the NormText column
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .text import Normalizer, NormalizerConfig


def _normalizer(args: argparse.Namespace) -> Normalizer:
    cfg = NormalizerConfig(numbers=args.numbers, g_letter=args.g_letter)
    return Normalizer(cfg, user_lexicon=args.user_lexicon)


def _lines(args: argparse.Namespace) -> list[str]:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8").splitlines()
    if args.text:
        return [" ".join(args.text)]
    return sys.stdin.read().splitlines()


def cmd_normalize(args: argparse.Namespace) -> None:
    norm = _normalizer(args)
    for line in _lines(args):
        print(norm(line) if line.strip() else "")


def cmd_unknown(args: argparse.Namespace) -> None:
    norm = _normalizer(args)
    counts: Counter[str] = Counter()
    for line in _lines(args):
        res = norm.analyze(line)
        counts.update(t.text.lower() for t in res.tokens if t.unknown)
    if not counts:
        print("No unknown Latin words.")
        return
    print("Kept in Latin. If a word is Derja, add it to your user lexicon (word<TAB>arabic).")
    print("If it's English/French, add it with the value '-' to silence it.\n")
    for word, n in counts.most_common():
        print(f"{n:5d}  {word}")


def cmd_metadata(args: argparse.Namespace) -> None:
    """Recompute column 3 (NormText) of an LJSpeech-style metadata.csv: id|raw|norm."""
    norm = _normalizer(args)
    path = Path(args.csv)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        clip_id, raw = parts[0], parts[1] if len(parts) > 1 else ""
        rows.append(f"{clip_id}|{raw}|{norm(raw)}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Updated {len(rows)} rows in {path}")


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--numbers", choices=["derja", "fr", "keep"], default="derja")
    common.add_argument("--g-letter", default="ڨ", help="Arabic letter for Arabizi 'g' (default ڨ)")
    common.add_argument("--user-lexicon", help="extra TSV lexicon (word<TAB>arabic, or '-' to keep Latin)")

    parser = argparse.ArgumentParser(prog="derja-tts", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("normalize", parents=[common], help="print normalized text")
    p.add_argument("text", nargs="*")
    p.add_argument("--file")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("unknown", parents=[common], help="list Latin words the lexicons don't know")
    p.add_argument("text", nargs="*")
    p.add_argument("--file")
    p.set_defaults(func=cmd_unknown)

    p = sub.add_parser("metadata", parents=[common], help="refresh NormText in metadata.csv")
    p.add_argument("csv")
    p.set_defaults(func=cmd_metadata)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
