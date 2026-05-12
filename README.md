# Granite Speech Demo

A web demo for [IBM Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — multilingual speech-to-text and speech translation running locally via HuggingFace Transformers.

## Features

- **Record** audio directly in the browser (start/stop button)
- **Transcribe** to English (always included)
- **Translate** to German, French, and/or Spanish (user-selectable)
- Live waveform visualisation while recording

## Requirements

- Python 3.10+
- ~5 GB disk for the model weights
- A GPU is strongly recommended; CPU will work but is slow

## Setup

```bash
pip install -r requirements.txt
```

On first run the model (~4.4 GB) is downloaded automatically from HuggingFace and cached in `~/.cache/huggingface/`.

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Notes

- The backend loads the model once at startup; subsequent requests are fast.
- Each selected output language requires a separate model pass, so selecting all four will take roughly 4x longer.
- Audio is resampled to mono 16 kHz before inference, matching the model's requirements.
