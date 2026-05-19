"""
Train a token classifier (NER-style) to detect German idioms.
"""

import json
import os
import random
from typing import List, Tuple

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
IDIOM_JSON = os.path.join(PROJECT_DIR, "data", "idiom_dictionary.json")
MODEL_OUT = os.path.join(PROJECT_DIR, "models", "idiom_detector")
BASE_MODEL = "bert-base-german-cased"


def load_idioms():
    with open(IDIOM_JSON, "r") as f:
        return json.load(f)


def create_synthetic_data(idioms: dict, n_samples: int = 500):
    """Create synthetic IOB-tagged data by injecting idioms into random contexts."""
    idiom_list = list(idioms.keys())
    data = []
    
    # Context templates
    templates = [
        "Er sagte, dass wir {idiom} sollten.",
        "Ich glaube, er will {idiom}.",
        "Warum müssen wir immer {idiom}?",
        "Es ist Zeit, dass sie {idiom}.",
        "Gestern haben wir {idiom}.",
        "Manchmal ist es besser, {idiom}.",
        "Sie hat entschieden, {idiom}.",
        "Kannst du bitte {idiom}?",
    ]

    for _ in range(n_samples):
        idiom = random.choice(idiom_list)
        template = random.choice(templates)
        sentence = template.format(idiom=idiom)
        
        # Find start/end of idiom in sentence
        start_idx = sentence.find(idiom)
        end_idx = start_idx + len(idiom)
        
        data.append({
            "text": sentence,
            "idiom": idiom,
            "start": start_idx,
            "end": end_idx
        })
    
    return data


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():
    idioms = load_idioms()
    raw_data = create_synthetic_data(idioms, n_samples=1000)
    
    # Convert to tokens and IOB tags
    formatted_data = []
    for item in raw_data:
        text = item["text"]
        idiom = item["idiom"]
        
        # Simple word splitting (better than nothing for synthetic)
        tokens = text.split()
        tags = [0] * len(tokens) # 0 = O
        
        idiom_tokens = idiom.split()
        for i in range(len(tokens) - len(idiom_tokens) + 1):
            if tokens[i:i+len(idiom_tokens)] == idiom_tokens:
                tags[i] = 1 # B-IDIOM
                for j in range(1, len(idiom_tokens)):
                    tags[i+j] = 2 # I-IDIOM
                break
        
        formatted_data.append({"tokens": tokens, "ner_tags": tags})

    # Train/Val split
    random.shuffle(formatted_data)
    split = int(0.8 * len(formatted_data))
    train_data = formatted_data[:split]
    val_data = formatted_data[split:]

    ds = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenized_ds = ds.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label={0: "O", 1: "B-IDIOM", 2: "I-IDIOM"}, label2id={"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
    )

    training_args = TrainingArguments(
        output_dir="./results_detector",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    print("Starting detector training...")
    trainer.train()
    
    os.makedirs(MODEL_OUT, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
