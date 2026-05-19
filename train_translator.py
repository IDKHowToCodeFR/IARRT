"""
Fine-tune mBART with LoRA for idiom-aware translation.
"""

import os
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
SUBSET_CSV = os.path.join(PROJECT_DIR, "data", "subset.csv")
IDIOM_JSON = os.path.join(PROJECT_DIR, "data", "idiom_dictionary.json")
MODEL_OUT = os.path.join(PROJECT_DIR, "models", "mbart_lora")
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"


def load_data():
    df = pd.read_csv(SUBSET_CSV)
    with open(IDIOM_JSON, "r") as f:
        idioms = pd.read_json(f, typ="series").to_dict()
    return df, idioms


def preprocess_function(examples, tokenizer):
    inputs = [f"German: {de} | Context: {ctx}" for de, ctx in zip(examples["de"], examples["context"])]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    labels = tokenizer(text_target=examples["en"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    df, idioms = load_data()
    
    # Create synthetic idiom-aware training samples
    idiom_data = []
    for de_idiom, en_meaning in idioms.items():
        # Simple templates for idiom-rich examples
        templates = [
            (f"Ich muss {de_idiom}.", f"I have to {en_meaning}."),
            (f"Er wollte {de_idiom}.", f"He wanted to {en_meaning}."),
            (f"Sie hat {de_idiom}.", f"She {en_meaning}."),
        ]
        for de, en in templates:
            idiom_data.append({"de": de, "en": en, "context": en_meaning})
    
    # Add some normal data with empty context
    normal_data = df.head(10).copy()
    normal_data["context"] = ""
    
    train_df = pd.concat([pd.DataFrame(idiom_data).head(10), normal_data], ignore_index=True)
    dataset = Dataset.from_pandas(train_df)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "en_XX"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_translator",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        save_total_limit=1,
        logging_steps=1,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print("Starting translator training (LoRA)...")
    trainer.train()

    os.makedirs(MODEL_OUT, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"LoRA weights saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
