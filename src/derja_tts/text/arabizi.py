"""Rule-based Arabizi -> Arabic script for Tunisian Derja.

Rules give a consonant skeleton. They get common shapes right
("3andi" -> عندي, "a7na" -> احنا, "na9ra" -> نقرا) and miss words whose
spelling hides a long vowel or an emphatic letter ("3lech" should be علاش,
the rules give علش). Those go in lexicons/arabizi.tsv, which always wins.
"""

from __future__ import annotations

import re

# Longest match first.
DIGRAPHS = {
    "ch": "ش", "sh": "ش", "kh": "خ", "gh": "غ", "th": "ث", "dh": "ذ",
    "ou": "و", "oo": "و", "aa": "ا", "ii": "ي",
}

DIGITS = {"2": "ء", "3": "ع", "5": "خ", "7": "ح", "8": "غ", "9": "ق"}

CONSONANTS = {
    "b": "ب", "t": "ت", "j": "ج", "d": "د", "r": "ر", "z": "ز", "s": "س",
    "f": "ف", "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه",
    "w": "و", "y": "ي", "v": "ف", "p": "ب", "c": "ك", "x": "كس",
}

SHORT_VOWELS = set("ae")   # dropped inside a word
LONG_VOWELS = {"i": "ي", "o": "و", "u": "و"}

ARABIZI_DIGITS = set(DIGITS)
_HAS_LETTER = re.compile(r"[a-z]")


def looks_like_arabizi(token: str) -> bool:
    """True when a Latin token uses Arabizi digits (3andi, a7na, 9ra)."""
    low = token.lower()
    digits = {c for c in low if c.isdigit()}
    return bool(digits) and digits <= ARABIZI_DIGITS and bool(_HAS_LETTER.search(low))


def _split_units(word: str) -> list[str]:
    units, i = [], 0
    while i < len(word):
        pair = word[i:i + 2]
        if pair in DIGRAPHS:
            units.append(pair)
            i += 2
        else:
            units.append(word[i])
            i += 1
    return units


def transliterate(word: str, g_letter: str = "ڨ") -> str:
    """Transliterate one Arabizi word. Assumes the word is lowercase-able Latin + digits."""
    units = _split_units(word.lower())
    out: list[str] = []
    last = len(units) - 1
    for i, u in enumerate(units):
        if u in DIGRAPHS:
            ar = DIGRAPHS[u]
            if u in ("ou", "oo") and i == 0:
                ar = "او"
        elif u in DIGITS:
            ar = DIGITS[u]
        elif u == "g":
            ar = g_letter
        elif u in CONSONANTS:
            ar = CONSONANTS[u]
        elif u in SHORT_VOWELS:
            if i == 0:
                ar = "ا"
            elif i == last and u == "a":
                ar = "ا"          # barcha -> برشا (use the lexicon for خدمة-type words)
            else:
                ar = ""           # short vowel, not written
        elif u in LONG_VOWELS:
            ar = "ا" if i == 0 else LONG_VOWELS[u]
        else:
            ar = ""               # apostrophes, accents, anything else
        # Collapse doubled letters: barra -> برا, famma -> فما
        if ar and out and out[-1] == ar and len(ar) == 1:
            continue
        out.append(ar)
    return "".join(out)
