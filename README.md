# derja-tts

Text-to-speech for Tunisian Derja. The first job: narrate tech videos in the voice of the person who wrote the script.

> **Status: early.** The text normalizer and the data pipeline work and are tested. Fine-tuning is the next milestone.

## Why

Commercial voice tools list "Arabic". Feed them a Tunisian line like `lyoum bech nfasserlkom el overfitting` and you get MSA with a stranger's accent, or noise.

There's a second problem, and it's on the text side. A real Tunisian script mixes three writing systems in one sentence: Arabic letters, Arabizi (`3andi`, `a7na`), and English or French terms. No TTS front-end reads that.

So this repo does two things:

1. **A normalizer** that turns mixed Derja scripts into one clean form a model can read.
2. **A training pipeline** to teach an open TTS model (Chatterbox Multilingual, MIT licence) Derja, then your own voice.

## How it fits together

```
your script                normalizer                     Chatterbox Multilingual ("ar" token)
"3aslema, lyoum bech  -->  "عسلامة, اليوم باش        -->   stage 1: fine-tuned on LinTO Derja
 nfasserlkom el             نفسرلكم الoverfitting"        stage 2: fine-tuned on your voice
 overfitting"                                                      |
                                                                   v
                                                             narration.wav
```

## The normalizer

| You write | The model reads |
|---|---|
| `3aslema, lyoum bech nfasserlkom chnowa howa el overfitting.` | `عسلامة, اليوم باش نفسرلكم شنوة هوا الoverfitting.` |
| `Learning rate = 0.01, w 3malna 50 epochs bel Adam optimizer.` | `Learning rate يساوي صفر virgule صفر واحد, وعملنا خمسين epochs بالAdam optimizer.` |
| `Fel 2026, el GPU wala a7sen men el CPU.` | `فال ألفين و ستة و عشرين, الGPU ولا أحسن من الCPU.` |
| `باش نفهمو الgradient descent` | `باش نفهمو الgradient descent` (already clean, left alone) |

The rules, in order:

- **Arabic script** stays as is. Diacritics and tatweel go.
- **Arabizi** becomes Arabic script. A lexicon of everyday Derja words comes first (`3lech` → علاش). Letter rules catch the rest (`3`→ع, `7`→ح, `9`→ق, `5`→خ, `ch`→ش, `g`→ڨ).
- **English and French terms** stay in Latin. So do acronyms (`GPU`, `CNN`) and tech tokens with digits (`fp16`, `gpt-4`, `3d`).
- **Numbers** are read in Derja (`2026` → ألفين و ستة و عشرين). Use `--numbers fr` for French or `--numbers keep` to leave digits alone.
- **Two-world words** like `la`, `ma`, `men`, `win` get decided by their neighbours.
- **Unknown Latin words** stay Latin and land in a report. You teach the normalizer from that report.

### Usage

```bash
pip install -e .

derja-tts normalize "Taw nchoufou el code fi Colab"
# توا نشوفو الcode في Colab

derja-tts normalize --file my_script.txt > my_script.norm.txt
derja-tts unknown --file my_script.txt        # words it kept in Latin, by frequency
```

```python
from derja_tts.text import Normalizer, NormalizerConfig

norm = Normalizer(NormalizerConfig(numbers="derja"), user_lexicon="my_words.tsv")
norm("ma habtetch el loss")
res = norm.analyze("ma habtetch el loss")
res.unknown   # ['habtetch']
```

### Teach it your words

Create a TSV file, one entry per line: `word<TAB>arabic`. Use `-` as the value to say "this is English or French, keep it Latin".

```
habtetch	هبطتش
tla3et	طلعت
deploy	-
```

Pass it with `--user-lexicon my_words.tsv`. When an entry is useful to everyone, add it to `src/derja_tts/text/lexicons/arabizi.tsv` and open a PR.

## Data

| Source | What | Licence | Used for |
|---|---|---|---|
| [LinTO audio](https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn) (LINAGORA) | ~93 h of transcribed Tunisian speech, many speakers, with code-switching | CC BY 4.0 | stage 1 |
| Your own recordings | 1–2 h of you narrating | yours | stage 2 |

**Not used, on purpose:** TV series, films, radio, other people's YouTube videos, or any voice whose owner didn't agree. The reasons and a consent template are in [docs/DATA_POLICY.md](docs/DATA_POLICY.md).

## Pipeline

### 1. Baseline, before any training

Two Colab notebooks run the same lines with your voice as reference:

- [`notebooks/01a_baseline_chatterbox.ipynb`](notebooks/01a_baseline_chatterbox.ipynb): Chatterbox Multilingual, zero-shot, raw vs normalized text.
- [`notebooks/01b_baseline_silma_f5.ipynb`](notebooks/01b_baseline_silma_f5.ipynb): an existing Derja fine-tune of F5-TTS, for comparison.

Each one ends with a score sheet. That sheet becomes the bar every checkpoint has to beat.

### 2. Record yourself

Follow [docs/RECORDING_GUIDE.md](docs/RECORDING_GUIDE.md), then cut the sessions into clips:

```bash
pip install -e ".[audio]"
python scripts/segment_audio.py recordings/ --out data/youssef --speaker youssef
```

This writes 4–12 s clips and a `metadata.csv` with draft transcripts from a Tunisian ASR model. The drafts contain mistakes. Fix column 2 by ear, then refresh the normalized column:

```bash
derja-tts metadata data/youssef/metadata.csv
```

### 3. Prepare LinTO

```bash
python scripts/prepare_linto.py --out data/linto
python scripts/prepare_linto.py --out data/linto_small --configs AmenyKH --max-hours 2   # quick test
```

### 4. Fine-tune (next milestone)

The plan copies what worked when the community added other languages to Chatterbox ([Tajik](https://huggingface.co/Tohirju/chatterbox-mtl-tajik), [Finnish](https://huggingface.co/RASMUS/Chatterbox-Finnish)):

- train only the T3 text-to-speech-token transformer; S3Gen and the voice encoder stay frozen
- keep the stock vocabulary and borrow the closest language token, `ar`
- stage 1: multi-speaker Derja (LinTO), few epochs, low learning rate
- stage 2: your voice, more epochs
- pick the checkpoint by round-trip WER, not by "last one"

Both data scripts write the LJSpeech layout these trainers expect: `id|raw text|normalized text`.

### 5. Evaluate

- Round-trip WER: generate a line, transcribe it with a Tunisian ASR model, compare with the input.
- The same listening sheet as the baseline: Tunisian-ness, tech terms, voice similarity.

## Roadmap

- [x] Mixed-script normalizer (Arabic, Arabizi, Latin), Derja numbers, tests
- [x] Clip segmentation + draft transcripts for your own recordings
- [x] LinTO → training clips, with attribution file
- [x] Baseline notebooks
- [ ] Baseline scores
- [ ] 1–2 h of personal recordings
- [ ] Stage 1 fine-tune on LinTO
- [ ] Stage 2 fine-tune on one voice
- [ ] Round-trip WER evaluation script
- [ ] Release weights on Hugging Face + a demo Space

## Models and licences

| Model / data | Licence | Role here |
|---|---|---|
| [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox) (Resemble AI) | MIT | base model for the release |
| [silma-tts-derja](https://huggingface.co/Ghazouaniwala/silma-tts-derja) (F5-TTS fine-tune) | CC BY-NC 4.0 | baseline comparison only |
| F5-TTS pretrained weights | CC BY-NC 4.0 | not used in the release |
| TunArTTS corpus | research use | not used |
| LinTO audio | CC BY 4.0 | training data, credited below |

## Responsible use

- Clone your own voice, or a voice whose owner agreed in writing. Nobody else's.
- Chatterbox adds an inaudible watermark (PerTh) to every output. Leave it on.
- On YouTube, cloning your own voice for voiceovers [doesn't need the AI disclosure](https://support.google.com/youtube/answer/14328491). Cloning someone else's voice does.

## Credits

LinTO datasets by LINAGORA:

```bibtex
@misc{naouara2025linto,
  title         = {LinTO Audio and Textual Datasets to Train and Evaluate Automatic Speech Recognition in Tunisian Arabic Dialect},
  author        = {Hedi Naouara and Jean-Pierre Lorr{\'e} and J{\'e}r{\^o}me Louradour},
  year          = {2025},
  eprint        = {2504.02604},
  archivePrefix = {arXiv}
}
```

Chatterbox by [Resemble AI](https://github.com/resemble-ai/chatterbox).

## Licence

Code: [MIT](LICENSE). Released weights will follow the base model (MIT) and carry the CC BY 4.0 attribution for LinTO.
