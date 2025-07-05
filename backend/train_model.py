from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

BASE_MODEL = "bert-base-uncased"
OUT_DIR    = Path(__file__).parent / "models" / "bert"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("go_emotions")
labels_list = ds["train"].features["labels"].feature.names
n_labels     = len(labels_list)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)

def preprocess(batch):
    enc = tok(batch["text"], truncation=True, padding=False)
    mh = np.zeros((len(batch["labels"]), n_labels), dtype=np.float32)
    for i, labs in enumerate(batch["labels"]):
        mh[i, labs] = 1.0
    enc["labels"] = mh.tolist()
    return enc

ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=["text", "labels"],
)

ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model    = AutoModelForSequenceClassification.from_pretrained(
               BASE_MODEL,
               num_labels=n_labels,
               problem_type="multi_label_classification",
           )
collator = DataCollatorWithPadding(tok, return_tensors="pt")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float() 
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_steps=200,
)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset= ds["validation"],
    data_collator=collator,
)

trainer.train()
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)

with open(OUT_DIR / "labels.json", "w") as f:
    json.dump(labels_list, f, indent=2)

print("Model saved to", OUT_DIR)
