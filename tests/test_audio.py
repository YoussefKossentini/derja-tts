from derja_tts.audio import group_speech, pad, safe_id, write_metadata


def test_short_regions_merge_until_max():
    regions = [(0, 3), (3.5, 6), (6.4, 9), (9.3, 14)]
    assert group_speech(regions, min_dur=4, max_dur=12) == [(0, 9), (9.3, 14)]


def test_long_pause_starts_new_clip():
    regions = [(0, 5), (7, 12)]
    assert group_speech(regions, max_gap=0.8) == [(0, 5), (7, 12)]


def test_too_short_clip_is_dropped():
    assert group_speech([(0, 2), (10, 11)], min_dur=4) == []


def test_long_region_is_split_evenly():
    clips = group_speech([(0, 30)], min_dur=4, max_dur=12)
    assert len(clips) == 3
    assert all(abs((e - s) - 10) < 1e-9 for s, e in clips)


def test_pad_is_clamped():
    assert pad((0.1, 9.95), 0.15, 10.0) == (0.0, 10.0)


def test_safe_id():
    assert safe_id("ep 01/intro.wav") == "ep_01_intro_wav"


def test_write_metadata(tmp_path):
    out = tmp_path / "metadata.csv"
    write_metadata(out, [("a_0001", "raw | text", "norm\ntext")])
    assert out.read_text(encoding="utf-8") == "a_0001|raw   text|norm text\n"
