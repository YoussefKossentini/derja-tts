"""Spell out numbers the way a Tunisian narrator would read them.

Three modes:
  - "derja": Tunisian number words in Arabic script (default)
  - "fr":    French number words (common in tech talk: "zéro virgule zéro un")
  - "keep":  leave digits untouched
"""

from __future__ import annotations

import re

UNITS = {
    0: "صفر", 1: "واحد", 2: "زوز", 3: "ثلاثة", 4: "أربعة", 5: "خمسة",
    6: "ستة", 7: "سبعة", 8: "ثمنية", 9: "تسعة", 10: "عشرة",
}
# Inside compounds Tunisians say "ثنين و عشرين", never "زوز و عشرين".
UNITS_IN_COMPOUND = {**UNITS, 2: "ثنين"}
TEENS = {
    11: "حداش", 12: "ثناش", 13: "ثلطاش", 14: "أربعطاش", 15: "خمسطاش",
    16: "سطاش", 17: "سبعطاش", 18: "ثمنطاش", 19: "تسعطاش",
}
TENS = {
    20: "عشرين", 30: "ثلاثين", 40: "أربعين", 50: "خمسين",
    60: "ستين", 70: "سبعين", 80: "ثمانين", 90: "تسعين",
}
HUNDREDS = {
    100: "مية", 200: "ميتين", 300: "ثلاثمية", 400: "أربعمية", 500: "خمسمية",
    600: "ستمية", 700: "سبعمية", 800: "ثمنمية", 900: "تسعمية",
}
AND = " و "

PERCENT_WORD = {"derja": "بالمية", "fr": "pour cent", "keep": "%"}

NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")


def is_number(token: str) -> bool:
    return bool(NUMBER_RE.match(token))


def _below_100(n: int) -> str:
    if n <= 10:
        return UNITS[n]
    if n < 20:
        return TEENS[n]
    tens, unit = divmod(n, 10)
    if unit == 0:
        return TENS[tens * 10]
    return f"{UNITS_IN_COMPOUND[unit]}{AND}{TENS[tens * 10]}"


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(HUNDREDS[hundreds * 100])
    if rest:
        parts.append(_below_100(rest))
    return AND.join(parts) if parts else UNITS[0]


def _scale(k: int, one: str, two: str, plural: str) -> str:
    """Read k thousands / k millions: ألف, ألفين, ثلاثة آلاف, حداش ألف..."""
    if k == 1:
        return one
    if k == 2:
        return two
    if 3 <= k <= 10:
        return f"{UNITS[k]} {plural}"
    return f"{_below_1000(k)} {one}"


def derja_int(n: int) -> str:
    if n < 0:
        return "ناقص " + derja_int(-n)
    if n < 1000:
        return _below_1000(n)
    if n >= 10**12:
        return " ".join(UNITS[int(d)] for d in str(n))
    millions, rest = divmod(n, 10**6)
    thousands, rest = divmod(rest, 1000)
    parts = []
    if millions:
        parts.append(_scale(millions, "مليون", "زوز ملاين", "ملاين"))
    if thousands:
        parts.append(_scale(thousands, "ألف", "ألفين", "آلاف"))
    if rest:
        parts.append(_below_1000(rest))
    return AND.join(parts)


def _digits(s: str, mode: str) -> str:
    return " ".join(spell_int(int(d), mode) for d in s)


def spell_int(n: int, mode: str) -> str:
    if mode == "fr":
        from num2words import num2words

        return num2words(n, lang="fr")
    return derja_int(n)


def spell_number(token: str, mode: str = "derja", decimal_word: str = "virgule") -> str:
    """Spell a numeric token like '25', '2026', '0.01' or '3,5'."""
    if mode == "keep":
        return token
    if not is_number(token):
        raise ValueError(f"not a number: {token!r}")
    int_part, _, frac = token.replace(",", ".").partition(".")

    # "007" reads digit by digit, "2026" reads as a number.
    if len(int_part) > 1 and int_part.startswith("0"):
        spoken = _digits(int_part, mode)
    else:
        spoken = spell_int(int(int_part), mode)

    if frac:
        # "0.01" -> "صفر virgule صفر واحد", "3.14" -> "ثلاثة virgule أربعطاش"
        if frac.startswith("0") or len(frac) > 2:
            frac_spoken = _digits(frac, mode)
        else:
            frac_spoken = spell_int(int(frac), mode)
        spoken = f"{spoken} {decimal_word} {frac_spoken}"
    return spoken
