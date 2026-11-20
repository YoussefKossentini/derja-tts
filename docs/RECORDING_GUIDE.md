# Recording guide

Goal: 1 to 2 hours of clean speech, in the voice you use for your videos.
One clean hour beats five noisy ones. The model copies everything it hears, including the fridge.

## Gear

- Any USB mic, or a recent phone. Use the **same mic for every session**.
- 15 to 20 cm from your mouth, slightly to the side so the "b" and "p" don't pop.
- Headphones to check the result.

## Room

- Small and soft. A wardrobe full of clothes is a great booth.
- Turn off the fan, AC and anything that hums. Close the window.
- Record 10 seconds of silence first and listen with headphones. If you hear something, so will the model.

## Settings

| Setting | Value |
|---|---|
| Format | WAV (not mp3) |
| Sample rate | 48 kHz or 44.1 kHz |
| Channels | mono |
| Level | peaks around -12 to -6 dBFS, never clipping |
| Effects | none: no noise suppression, no compressor, no music |

Write the settings down. Next session, same settings.

## What to read

1. **Your own past scripts.** That's the exact domain the model has to learn: Derja with English tech terms.
2. **New tech lines** with the vocabulary you'll use later (attention, embeddings, loss, dataset...).
3. **General Derja sentences**, for example from the [LinTO text dataset](https://huggingface.co/datasets/linagora/linto-dataset-text-ar-tn), so the model doesn't only know tech talk.

Read the way you narrate. Not like a news anchor. Leave a clear pause (half a second or more) between sentences, because the segmenter cuts there.

## Sessions

- 20 to 30 minutes, then a break. Your voice changes when you're tired.
- Same time of day if you can. Warm up for two minutes, drink water.
- Messed up a sentence? Pause, then read the whole sentence again. Delete the bad clip later.

## Old videos

Your existing narrations are perfect data **if there was no music under the voice**.
Extract the audio:

```bash
ffmpeg -i video.mp4 -vn -ac 1 -ar 48000 narration.wav
```

If music was mixed in, skip that video. Vocal separation leaves artifacts and the model learns them.

## After recording

```bash
python scripts/segment_audio.py recordings/ --out data/youssef --speaker youssef
```

1. Open `data/youssef/metadata.csv`. Column 2 is the ASR draft.
2. Listen to each clip and fix column 2. Write it the way you write your scripts (mixed is fine).
3. Delete rows and wavs with coughs, laughs or mistakes.
4. Refresh the normalized column: `derja-tts metadata data/youssef/metadata.csv`
5. Run `derja-tts unknown --file <(cut -d'|' -f2 data/youssef/metadata.csv)` and add any Derja word it lists to your user lexicon.
