# Data policy

A voice model learns two things from its data: a language and the voices in it. So "where does the audio come from" decides what the model can legally do and whose voice it can produce.

## Allowed

- **Your own voice.** You recorded it, you own it.
- **People who agreed in writing.** Friends or family who read prompts for the project, with the consent form below signed.
- **Datasets whose licence allows training and redistribution**, credited as the licence asks. Today that's the LinTO audio dataset (CC BY 4.0).

## Not allowed

- TV series, films, sketches, radio shows, podcasts, songs.
- Other people's YouTube or TikTok videos.
- Any recording of a person who didn't agree, including public figures.

Why, in short:

1. **Copyright.** Broadcasts belong to channels and producers. No licence covers training a model on them and publishing it.
2. **Consent.** A model trained on an actor's lines can produce new sentences in that actor's voice. That's cloning a real person without asking.
3. **Takedowns.** YouTube lets anyone request removal of AI content that simulates an identifiable person's voice. One request can take down the video. A complaint to GitHub or Hugging Face can take down the repo.
4. **Quality.** Series audio has music, effects, overlapping lines and no transcripts. It makes poor TTS data anyway.

## Before publishing weights

- The model card lists every data source and its licence.
- `attribution.csv` from `prepare_linto.py` ships with the training config.
- Speakers other than you have a signed consent form on file.
- The PerTh watermark stays on in the demo.

## Consent form template

Translate it into Derja or French if that's easier for the speaker. Keep one signed copy per person.

```
Voice recording consent: derja-tts

I, ______________________, agree that the recordings of my voice made on
____________ (dates) may be used to train the derja-tts speech model, and that
the trained model weights and audio samples may be published publicly under
the ______ licence.

I understand that:
- the model can generate new speech that sounds like my voice;
- I can withdraw my consent for future releases by contacting
  ______________________; models already published cannot be recalled.

Signature: ____________________     Date: ____________
```
