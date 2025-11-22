# lmdk/train.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import os
from rich.console import Console
import time
import mlflow
from accelerate import Accelerator

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
        if self.use_mlflow:
            mlflow.start_run()
            self.run_id = mlflow.active_run().info.run_id
            mlflow.log_params(config)
        console.print(f"[bold blue]Experiment started:[/] {self.experiment_name}")

    def log_metric(self, key: str, value, step: int = None):
        if self.use_mlflow:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int = None):
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

    def end_experiment(self):
        if self.start_time:
            duration = time.time() - self.start_time
            self.log_metric('training_time', duration)
            console.print(f"[bold green]Experiment completed in {duration:.2f} seconds[/]")
            if self.use_mlflow:
                mlflow.end_run()
                console.print(f"[dim]MLflow run ID: {self.run_id}[/]")

def train_model(
    model_name: str = "gpt2",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
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
    Hugging Face 모델을 사용하여 텍스트 생성 모델을 훈련합니다.
    """
    console.print(f"[bold blue]모델 훈련 시작:[/] {model_name}")
    console.print(f"[bold blue]데이터셋:[/] {dataset_name} ({dataset_config})")

    # Accelerator 초기화 (distributed training)
    accelerator = Accelerator() if use_accelerate else None
    if accelerator:
        console.print(f"[bold yellow]Accelerate 사용:[/] {accelerator.num_processes} 프로세스")

    # Telemetry 초기화
    tracker = TelemetryTracker() if use_telemetry else None

    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "num_train_epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "use_accelerate": use_accelerate,
    }

    if tracker:
        tracker.start_experiment(config)

    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Accelerator로 모델 래핑
    if accelerator:
        model = accelerator.prepare(model)

    # 데이터셋 로드
    dataset = load_dataset(dataset_name, dataset_config)

    # 토크나이징 함수
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 훈련 인자 설정
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
        dataloader_num_workers=0,  # Windows 호환성
    )

    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
    )

    # 훈련 시작
    console.print("[bold green]훈련 시작...[/]")
    train_result = trainer.train()

    if tracker:
        tracker.log_metrics({
            "final_loss": train_result.training_loss,
            "num_train_samples": len(tokenized_datasets["train"]),
        })

    # 모델 저장
    if accelerator:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
    else:
        trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    console.print(f"[bold green]훈련 완료! 모델이 저장되었습니다:[/] {output_dir}")

    if tracker:
        tracker.end_experiment()

    return trainer