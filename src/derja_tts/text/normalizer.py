"""Turn a mixed Derja script into text a TTS model can read.

Input, the way Tunisians actually type a video script:
    "3aslema, lyoum bech nfasserlkom chnowa howa el overfitting."
    "باش نفهمو الgradient descent, learning rate = 0.01"

Output, one canonical form:
    - Derja in Arabic script (Arabizi gets transliterated)
    - English / French terms kept in Latin script
    - numbers and symbols spelled out
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .arabizi import looks_like_arabizi, transliterate
from .numbers import PERCENT_WORD, is_number, spell_number

# ---------------------------------------------------------------- tokenizer
_AR = r"[ؠ-ؿف-ٰٟٮ-ۓەۺ-ۿݐ-ݿ]+"
_LAT_CHARS = r"A-Za-z0-9À-ÖØ-öø-ÿ"
_LAT = rf"[{_LAT_CHARS}]+(?:['’.\-][{_LAT_CHARS}]+)*"
TOKEN_RE = re.compile(rf"(?P<ar>{_AR})|(?P<lat>{_LAT})|(?P<p>\S)")

DIACRITICS_RE = re.compile(r"[ً-ٰٟ]")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_NUM_PUNCT = {"٪": "%", "٫": ".", "٬": ""}

UNIT_LIKE = re.compile(r"^\d+[a-z]{1,2}$")              # 3d, 4k, 16gb
DIM_LIKE = re.compile(r"^\d+x\d+$")                      # 3x3
VERSION_LIKE = re.compile(r"^[a-z]+[-.]\d+(\.\d+)*$|^[a-z]*\d+\.\d+|^[a-z]\d+$")  # gpt-4, v3.5, v2
FRENCH_ELISION = re.compile(r"^(l|d|c|j|n|m|t|s|qu)['’]")

KEEP_PUNCT = set(".,!?;:،؟؛…")
SENTENCE_END = set(".!?؟…")
SYMBOLS = {
    "derja": {"=": "يساوي", "+": "زايد", "%": PERCENT_WORD["derja"]},
    "fr": {"=": "égal", "+": "plus", "%": PERCENT_WORD["fr"]},
    "keep": {"=": "=", "+": "+", "%": "%"},
}

ARABIC_KINDS = {"ARABIC", "ARABIZI"}


@dataclass
class NormalizerConfig:
    numbers: str = "derja"          # "derja" | "fr" | "keep"
    decimal_word: str = "virgule"   # how "0.01" sounds: "صفر virgule صفر واحد"
    g_letter: str = "ڨ"             # Arabizi "g". Switch to "ق" if your TTS vocab lacks ڨ
    strip_diacritics: bool = True
    glue_article_to_latin: bool = True  # "el model" -> "الmodel" (False -> "ال model")


@dataclass
class Token:
    text: str
    kind: str                # ARABIC | ARABIZI | LATIN | NUMBER | PUNCT | AMBIG
    space_before: bool
    out: str = ""
    glue_next: bool = False
    unknown: bool = False


@dataclass
class Result:
    text: str
    unknown: list[str] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)


def _read_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def _read_tsv(text: str) -> dict[str, str]:
    table = {}
    for ln in _read_lines(text):
        key, _, value = ln.partition("\t")
        if value:
            table[key.strip().lower()] = value.strip()
    return table


def _builtin(name: str) -> str:
    return resources.files("derja_tts.text").joinpath("lexicons", name).read_text(encoding="utf-8")


class Normalizer:
    """Callable: Normalizer()(text) -> normalized text. Use .analyze() for details."""

    def __init__(self, config: NormalizerConfig | None = None, user_lexicon: str | Path | None = None):
        self.config = config or NormalizerConfig()
        if self.config.numbers not in SYMBOLS:
            raise ValueError("numbers must be 'derja', 'fr' or 'keep'")
        self.arabizi = _read_tsv(_builtin("arabizi.tsv"))
        self.ambiguous = _read_tsv(_builtin("ambiguous.txt"))
        self.keep_latin = {w.lower() for w in _read_lines(_builtin("keep_latin.txt"))}
        if user_lexicon:
            # Same TSV format. A value of "-" means "keep this word in Latin".
            for key, value in _read_tsv(Path(user_lexicon).read_text(encoding="utf-8")).items():
                self.ambiguous.pop(key, None)
                if value == "-":
                    self.keep_latin.add(key)
                    self.arabizi.pop(key, None)
                else:
                    self.arabizi[key] = value
                    self.keep_latin.discard(key)

    # ------------------------------------------------------------ public API
    def __call__(self, text: str) -> str:
        return self.analyze(text).text

    def analyze(self, text: str) -> Result:
        tokens = self._tokenize(self._clean(text))
        for tok in tokens:
            self._classify(tok)
        self._resolve_ambiguous(tokens)
        self._render(tokens)
        unknown = list(dict.fromkeys(t.text.lower() for t in tokens if t.unknown))
        return Result(text=self._join(tokens), unknown=unknown, tokens=tokens)

    # ------------------------------------------------------------ steps
    def _clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(ARABIC_DIGITS)
        for src, dst in ARABIC_NUM_PUNCT.items():
            text = text.replace(src, dst)
        text = text.replace("ـ", "")  # tatweel
        if self.config.strip_diacritics:
            text = DIACRITICS_RE.sub("", text)
        return text

    @staticmethod
    def _tokenize(text: str) -> list[Token]:
        tokens = []
        for m in TOKEN_RE.finditer(text):
            kind = "ARABIC" if m.group("ar") else "LATIN" if m.group("lat") else "PUNCT"
            space = m.start() > 0 and text[m.start() - 1].isspace()
            tokens.append(Token(m.group(0), kind, space))
        return tokens

    def _classify(self, tok: Token) -> None:
        if tok.kind != "LATIN":
            return
        low = tok.text.lower()
        if is_number(tok.text):
            tok.kind = "NUMBER"
        elif low in self.keep_latin:
            tok.kind = "LATIN"
        elif low in self.ambiguous:
            tok.kind = "AMBIG"
        elif low in self.arabizi:
            tok.kind = "ARABIZI"
        elif UNIT_LIKE.match(low) or DIM_LIKE.match(low) or VERSION_LIKE.match(low):
            tok.kind = "LATIN"
        elif FRENCH_ELISION.match(low):
            tok.kind = "LATIN"
        elif tok.text.isupper() and tok.text.isalpha() and len(tok.text) >= 2:
            tok.kind = "LATIN"  # CNN, GPU, API
        elif looks_like_arabizi(low):
            tok.kind = "ARABIZI"
        else:
            tok.kind = "LATIN"
            tok.unknown = True  # kept in Latin, but worth a look

    @staticmethod
    def _neighbours(tokens: list[Token], i: int, step: int) -> str | None:
        j = i + step
        while 0 <= j < len(tokens):
            t = tokens[j]
            if t.kind == "PUNCT":
                if t.text in SENTENCE_END:
                    return None
            elif t.kind != "NUMBER":
                return t.kind
            j += step
        return None

    def _resolve_ambiguous(self, tokens: list[Token]) -> None:
        for i, tok in enumerate(tokens):
            if tok.kind != "AMBIG":
                continue
            sides = [self._neighbours(tokens, i, -1), self._neighbours(tokens, i, 1)]
            derja = sum(s in ARABIC_KINDS or s == "AMBIG" for s in sides)
            latin = sum(s == "LATIN" for s in sides)
            tok.kind = "LATIN" if latin > derja else "ARABIZI"

    def _render(self, tokens: list[Token]) -> None:
        cfg = self.config
        symbols = SYMBOLS[cfg.numbers]
        skip_next = False
        for i, tok in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if tok.kind == "ARABIC":
                tok.out = tok.text
            elif tok.kind == "ARABIZI":
                low = tok.text.lower()
                value = self.arabizi.get(low) or self.ambiguous.get(low)
                if value is None:
                    value = transliterate(low, g_letter=cfg.g_letter)
                if value.endswith("+"):
                    tok.glue_next = True
                    value = value[:-1]
                tok.out = value
            elif tok.kind == "NUMBER":
                tok.out = spell_number(tok.text, cfg.numbers, cfg.decimal_word)
                if nxt is not None and nxt.text == "%":
                    tok.out = f"{tok.out} {symbols['%']}"
                    skip_next = True
            elif tok.kind == "LATIN":
                tok.out = tok.text
            elif tok.kind == "PUNCT":
                if tok.text in KEEP_PUNCT:
                    tok.out = tok.text
                elif tok.text in symbols:
                    tok.out = symbols[tok.text]
                else:
                    tok.out = ""  # quotes, markdown, slashes...

    def _join(self, tokens: list[Token]) -> str:
        parts: list[str] = []
        prev: Token | None = None
        prev_index = -2
        for i, tok in enumerate(tokens):
            if not tok.out:
                continue
            out = tok.out
            if prev is None:
                parts.append(out)
            elif tok.kind == "PUNCT" and tok.out in KEEP_PUNCT:
                parts.append(out)
            elif prev.glue_next and (
                tok.kind in ARABIC_KINDS
                or (self.config.glue_article_to_latin and tok.kind == "LATIN" and prev.out.endswith(("ال", "لل")))
            ):
                if prev.out.endswith("ال") and out.startswith("ال"):
                    out = out[2:]  # "fel lyoum" -> فاليوم, not فالاليوم
                parts.append(out)
            elif not tok.space_before and prev_index == i - 1 and prev.kind != "PUNCT" and tok.kind != "PUNCT":
                parts.append(out)  # script switch inside one word: "الmodel"
            else:
                parts.append(" " + out)
            prev, prev_index = tok, i
        return re.sub(r"\s+", " ", "".join(parts)).strip()
