from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import os
from fastapi.responses import FileResponse

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model_summary")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)
tokenizer = T5Tokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

@app.get("/")
async def home():
    return FileResponse(INDEX_PATH)

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}