import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "ibm-granite/granite-speech-4.1-2b"
TARGET_SR = 16000

LANGUAGE_PROMPTS = {
    "english": "transcribe the speech with proper punctuation and capitalization.",
    "german": "translate the speech to German with proper punctuation and capitalization.",
    "french": "translate the speech to French with proper punctuation and capitalization.",
    "spanish": "translate the speech to Spanish with proper punctuation and capitalization.",
}

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Granite Speech model: %s", MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, device_map=device, torch_dtype=torch.bfloat16
    )
    model.eval()

    model_state["processor"] = processor
    model_state["tokenizer"] = processor.tokenizer
    model_state["model"] = model
    model_state["device"] = device
    logger.info("Model loaded successfully")
    yield
    model_state.clear()


app = FastAPI(title="Granite Speech Demo", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def load_audio(data: bytes) -> torch.Tensor:
    """Load audio bytes, convert to mono 16kHz float tensor."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        wav, sr = torchaudio.load(tmp_path, normalize=True)
    finally:
        os.unlink(tmp_path)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        wav = resampler(wav)

    return wav


def run_inference(wav: torch.Tensor, task_prompt: str) -> str:
    processor = model_state["processor"]
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    device = model_state["device"]

    user_prompt = f"<|audio|>{task_prompt}"
    chat = [{"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)

    with torch.inference_mode():
        model_outputs = model.generate(
            **model_inputs, max_new_tokens=400, do_sample=False, num_beams=1
        )

    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
    text = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
    return text[0].strip()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
async def transcribe(
    audio: Annotated[UploadFile, File()],
    languages: Annotated[str, Form()] = "english",
):
    if not model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        wav = load_audio(audio_bytes)
    except Exception as e:
        logger.exception("Failed to load audio")
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    requested = [lang.strip().lower() for lang in languages.split(",") if lang.strip()]
    # English is always included
    if "english" not in requested:
        requested.insert(0, "english")

    results = {}
    for lang in requested:
        prompt = LANGUAGE_PROMPTS.get(lang)
        if prompt is None:
            continue
        try:
            results[lang] = run_inference(wav, prompt)
        except Exception as e:
            logger.exception("Inference failed for language: %s", lang)
            results[lang] = f"[Error: {e}]"

    return {"results": results}
