"""
Train a token classifier (NER-style) to detect German idioms.
"""

import json
import os
import random
import re
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
    if os.path.exists(IDIOM_JSON):
        with open(IDIOM_JSON, "r") as f:
            return json.load(f)
    from data.load_data import EXPANDED_IDIOMS
    return EXPANDED_IDIOMS


def normalize_text(text: str) -> str:
    text = text.casefold()
    replacements = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def create_synthetic_data(idioms: dict, n_samples: int = 1000):
    idiom_list = list(idioms.keys())
    data = []
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
        data.append({"text": sentence, "idiom": idiom})
    
    return data


def tokenize_and_align_labels(examples, tokenizer):
    # This function expects 'tokens' and 'ner_tags' in examples
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
                label_ids.append(label[word_idx]) # Subwords get same tag
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    idioms = load_idioms()
    raw_data = create_synthetic_data(idioms, n_samples=2000)
    
    formatted_data = []
    for item in raw_data:
        text = item["text"]
        idiom = item["idiom"]
        
        # Proper word tokenization for IOB alignment
        # We split by spaces but preserve punctuation as separate words
        words = re.findall(r"[\w']+|[.,!?;]", text)
        tags = [0] * len(words)
        
        idiom_words = re.findall(r"[\w']+|[.,!?;]", idiom)
        
        # Match idiom words (case insensitive)
        for i in range(len(words) - len(idiom_words) + 1):
            match = True
            for j in range(len(idiom_words)):
                if normalize_text(words[i+j]) != normalize_text(idiom_words[j]):
                    match = False
                    break
            if match:
                tags[i] = 1 # B-IDIOM
                for k in range(1, len(idiom_words)):
                    tags[i+k] = 2 # I-IDIOM
                break
        
        formatted_data.append({"tokens": words, "ner_tags": tags})

    random.shuffle(formatted_data)
    split = int(0.9 * len(formatted_data))
    train_data = formatted_data[:split]
    val_data = formatted_data[split:]

    ds = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    tokenized_ds = ds.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=3, 
        id2label={0: "O", 1: "B-IDIOM", 2: "I-IDIOM"}, 
        label2id={"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
    )

    training_args = TrainingArguments(
        output_dir="./results_detector",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        num_train_epochs=5,
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

    print("Starting robust detector training...")
    trainer.train()
    
    os.makedirs(MODEL_OUT, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
