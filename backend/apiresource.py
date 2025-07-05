from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from pathlib import Path
from typing import List

MODEL_DIR = Path(__file__).parent / "models" / "bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(MODEL_DIR / "labels.json", "r") as f:
    LABELS: List[str] = json.load(f)

app = FastAPI()

class TextIn(BaseModel):
    text: str

class EmotionsOut(BaseModel):
    emotions: List[str]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # or your specific origins
    allow_methods=["*"],            # includes OPTIONS, GET, POST, etc.
    allow_headers=["*"],            # needed if you send custom headers
    allow_credentials=True,
)

@app.post("/predict", response_model=EmotionsOut)
def predict(input: TextIn):
    # tokenize + predict
    inputs = tokenizer(
        input.text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.sigmoid(logits)[0].tolist()

    picked = [LABELS[i] for i, p in enumerate(probs) if p >= 0.5]

    return {"emotions": picked}
