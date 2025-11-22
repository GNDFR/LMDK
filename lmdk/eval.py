# lmdk/eval.py
"""
Evaluation Engine for standard benchmarks
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import lm_eval
from lm_eval import evaluator
from rich.console import Console
import json
import os

console = Console()

class EvaluationEngine:
    def __init__(self, model_path: str, model_name: str = None):
        self.model_path = model_path
        self.model_name = model_name or "hf-causal-experimental"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer"""
        console.print(f"[bold blue]Loading model from:[/] {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run_benchmark(self, tasks: list = None, num_fewshot: int = 0):
        """
        Run evaluation on specified benchmarks
        """
        if tasks is None:
            tasks = ["hellaswag", "winogrande", "piqa"]  # Common benchmarks

        if not self.model:
            self.load_model()

        console.print(f"[bold blue]Running benchmarks:[/] {', '.join(tasks)}")

        # Create lm_eval model wrapper
        model = lm_eval.models.huggingface.HFLM(
            pretrained=self.model_path,
            tokenizer=self.tokenizer,
            model_type="causal",
        )

        results = {}

        for task in tasks:
            console.print(f"[bold yellow]Evaluating {task}...[/]")
            try:
                result = lm_eval.evaluate(
                    model=model,
                    tasks=[task],
                    num_fewshot=num_fewshot,
                    limit=None,
                )
                results[task] = result
                console.print(f"[green]✓ {task} completed[/]")
            except Exception as e:
                console.print(f"[red]✗ {task} failed: {e}[/]")
                results[task] = {"error": str(e)}

        return results

    def save_results(self, results: dict, output_path: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        console.print(f"[bold green]Results saved to:[/] {output_path}")

def evaluate_model(
    model_path: str,
    tasks: list = None,
    num_fewshot: int = 0,
    output_path: str = "evaluation_results.json"
):
    """
    Convenience function to evaluate a model
    """
    engine = EvaluationEngine(model_path)
    results = engine.run_benchmark(tasks, num_fewshot)
    engine.save_results(results, output_path)
    return results