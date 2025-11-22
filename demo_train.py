#!/usr/bin/env python3
"""
Simple demo script to train a small LM using LMDK
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

def create_demo_dataset():
    """Create a small demo dataset from sample_data.txt"""
    with open('sample_data.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()

    # Filter out short lines and comments
    texts = [line.split('#')[0].strip() for line in texts if len(line.split('#')[0].strip()) > 10]

    return Dataset.from_dict({"text": texts})

def main():
    print("LMDK Demo: Training a small language model")

    # Load model and tokenizer
    model_name = "distilgpt2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create dataset
    print("Creating demo dataset...")
    dataset = create_demo_dataset()
    print(f"Dataset size: {len(dataset)} examples")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./demo_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        save_steps=50,
        logging_steps=10,
        save_total_limit=2,
        eval_strategy="no",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    trainer.save_model("./demo_output")
    tokenizer.save_pretrained("./demo_output")

    print("Training completed! Model saved to ./demo_output")

    # Test generation
    print("\nTesting text generation...")
    model.eval()
    input_text = "This is a test"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()