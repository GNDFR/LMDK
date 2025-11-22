# lmdk/train.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import os
from rich.console import Console
import time
from typing import Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

console = Console()

# Telemetry Tracker with MLflow integration
class TelemetryTracker:
    def __init__(self, experiment_name: str = "lmdk_training", use_mlflow: bool = True):
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.start_time = None
        self.run_id = None

        if self.use_mlflow:
            mlflow.set_experiment(self.experiment_name)

    def start_experiment(self, config: dict):
        self.start_time = time.time()
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.start_run()
            self.run_id = mlflow.active_run().info.run_id
            mlflow.log_params(config)
        console.print(f"[bold blue]Experiment started:[/] {self.experiment_name}")

    def log_metric(self, key: str, value, step: int = None):
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int = None):
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)

    def end_experiment(self):
        if self.start_time:
            duration = time.time() - self.start_time
            self.log_metric('training_time', duration)
            console.print(f"[bold green]Experiment completed in {duration:.2f} seconds[/]")
            if self.use_mlflow and MLFLOW_AVAILABLE:
                mlflow.end_run()
                console.print(f"[dim]MLflow run ID: {self.run_id}[/]")

def train_model(
    model_name: str = "gpt2",
    dataset_name: Optional[str] = "wikitext",
    dataset_config: Optional[str] = "wikitext-2-raw-v1",
    train_file: Optional[str] = None,
    output_dir: str = "./output",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 5e-5,
    save_steps: int = 500,
    logging_steps: int = 100,
    use_telemetry: bool = True,
    use_accelerate: bool = False,
):
    """
    Trains a text generation model using Hugging Face models.
    """
    console.print(f"[bold blue]Starting model training for:[/] {model_name}")
    if train_file:
        console.print(f"[bold blue]Training file:[/] {train_file}")
    else:
        console.print(f"[bold blue]Dataset:[/] {dataset_name} ({dataset_config})")

    # Initialize Accelerator (distributed training)
    accelerator = Accelerator() if use_accelerate and ACCELERATE_AVAILABLE else None
    if accelerator:
        console.print(f"[bold yellow]Using Accelerate with:[/] {accelerator.num_processes} processes")

    # Initialize Telemetry
    tracker = TelemetryTracker() if use_telemetry else None

    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "train_file": train_file,
        "num_train_epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "use_accelerate": use_accelerate,
    }

    if tracker:
        tracker.start_experiment(config)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Wrap model with Accelerator
    if accelerator and ACCELERATE_AVAILABLE:
        model = accelerator.prepare(model)

    # Load dataset
    if train_file:
        dataset = load_dataset("text", data_files={"train": train_file})
    else:
        dataset = load_dataset(dataset_name, dataset_config)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=2,
        evaluation_strategy="steps" if "validation" in tokenized_datasets else "no",
        eval_steps=save_steps if "validation" in tokenized_datasets else None,
        load_best_model_at_end=True if "validation" in tokenized_datasets else False,
        dataloader_num_workers=0,  # Windows compatibility
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
    )

    # Start training
    console.print("[bold green]Starting training...[/]")
    train_result = trainer.train()

    if tracker:
        tracker.log_metrics({
            "final_loss": train_result.training_loss,
            "num_train_samples": len(tokenized_datasets["train"]),
        })

    # Save model
    if accelerator and ACCELERATE_AVAILABLE:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
    else:
        trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    console.print(f"[bold green]Training complete! Model saved to:[/] {output_dir}")

    if tracker:
        tracker.end_experiment()

    return trainer