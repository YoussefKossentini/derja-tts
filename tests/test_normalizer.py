import pytest

from derja_tts.text import Normalizer, NormalizerConfig, looks_like_arabizi, spell_number, transliterate


@pytest.fixture(scope="module")
def norm():
    return Normalizer()


# --------------------------------------------------------------- arabizi rules
@pytest.mark.parametrize("word, expected", [
    ("3andi", "عندي"),
    ("a7na", "احنا"),
    ("na9ra", "نقرا"),
    ("ya3ni", "يعني"),
    ("barra", "برا"),
    ("ye5demch", "يخدمش"),
])
def test_transliterate_rules(word, expected):
    assert transliterate(word) == expected


def test_g_letter_is_configurable():
    assert transliterate("gal") == "ڨل"
    assert transliterate("gal", g_letter="ق") == "قل"


@pytest.mark.parametrize("token, expected", [
    ("3andi", True), ("mta3", True), ("so2al", True),
    ("fp16", False), ("gpt4", False), ("model", False), ("2026", False),
])
def test_looks_like_arabizi(token, expected):
    assert looks_like_arabizi(token) is expected


# --------------------------------------------------------------- numbers
@pytest.mark.parametrize("token, expected", [
    ("0", "صفر"),
    ("2", "زوز"),
    ("12", "ثناش"),
    ("22", "ثنين و عشرين"),
    ("50", "خمسين"),
    ("125", "مية و خمسة و عشرين"),
    ("2026", "ألفين و ستة و عشرين"),
    ("3000", "ثلاثة آلاف"),
    ("15000", "خمسطاش ألف"),
    ("0.01", "صفر virgule صفر واحد"),
    ("3.14", "ثلاثة virgule أربعطاش"),
    ("007", "صفر صفر سبعة"),
])
def test_derja_numbers(token, expected):
    assert spell_number(token) == expected


def test_french_numbers():
    assert spell_number("0.01", mode="fr") == "zéro virgule zéro un"
    assert spell_number("71", mode="fr") == "soixante et onze"


# --------------------------------------------------------------- full sentences
def test_arabizi_sentence(norm):
    assert norm("3aslema, lyoum bech nfasserlkom chnowa howa el overfitting.") == \
        "عسلامة, اليوم باش نفسرلكم شنوة هوا الoverfitting."


def test_arabic_script_with_latin_inside_word_is_preserved(norm):
    assert norm("باش نفهمو الgradient descent") == "باش نفهمو الgradient descent"


def test_article_glues_to_next_arabic_word_without_double_al(norm):
    assert norm("fel lyoum") == "فاليوم"


def test_article_glue_to_latin_can_be_turned_off():
    n = Normalizer(NormalizerConfig(glue_article_to_latin=False))
    assert n("el model") == "ال model"


def test_numbers_symbols_and_percent(norm):
    assert norm("rate = 0.01") == "rate يساوي صفر virgule صفر واحد"
    assert norm("92%") == "ثنين و تسعين بالمية"


def test_tech_tokens_with_digits_stay_latin(norm):
    out = norm("el fp16 w el 3d w gpt-4 w 16gb")
    for term in ("fp16", "3d", "gpt-4", "16gb"):
        assert term in out


def test_acronyms_stay_latin(norm):
    assert norm("el GPU a7sen men el CPU") == "الGPU أحسن من الCPU"


def test_particle_between_latin_words_is_still_derja(norm):
    assert norm("50 epochs bel Adam optimizer") == "خمسين epochs بالAdam optimizer"


def test_ambiguous_word_follows_context(norm):
    assert norm("ena ma fhemtech") .startswith("أنا ما")
    assert norm("la vie est belle") == "la vie est belle"


def test_unknown_latin_words_are_reported(norm):
    res = norm.analyze("ma habtetch el loss")
    assert res.unknown == ["habtetch"]


def test_diacritics_tatweel_and_arabic_digits(norm):
    assert norm("بَرْشَا ـــ ٣") == "برشا ثلاثة"


def test_user_lexicon_overrides(tmp_path):
    lex = tmp_path / "me.tsv"
    lex.write_text("habtetch\tهبطتش\nmodel\tموديل\nbarcha\t-\n", encoding="utf-8")
    n = Normalizer(user_lexicon=lex)
    assert n("ma habtetch") == "ما هبطتش"
    assert n("el model") == "الموديل"
    assert n("barcha") == "barcha"


def test_markdown_noise_is_dropped(norm):
    assert norm("**el model** (v2)") == "الmodel v2"
