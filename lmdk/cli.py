# lmdk/cli.py
import sys
import os

# Add project root to sys.path for ModuleNotFoundError resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated
from typing import Optional

# Import rust_core library
from rust_core import DataCleanser
import subprocess

try:
    from .train import train_model
    TRAIN_AVAILABLE = True
except ImportError:
    TRAIN_AVAILABLE = False

try:
    from .eval import evaluate_model
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

app = typer.Typer(
    name="lmdk",
    help="Language Model Development Kit - From data pipeline to model training",
    add_completion=False,
)

console = Console()

def _run_command(command, description):
    """Helper function to run a command and handle errors."""
    console.print(f"[dim]{description}...[/]")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Failed to {description.lower()}:[/] {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError:
        console.print(f"[bold red]{command[0]} is not installed. Please install it to continue.[/]")
        raise typer.Exit(code=1)

@app.command()
def prep(
    filepath: Annotated[
        str,
        typer.Argument(
            help="Path to the text file to process",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    min_length: Annotated[
        int, typer.Option(help="Minimum length to consider a sentence valid")
    ] = 20,
    toxic_keywords_file: Annotated[
        str,
        typer.Option(
            help="File with a list of toxic keywords to filter (one keyword per line)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    output_file: Annotated[
        Optional[str],
        typer.Option(
            help="Path to save the cleaned data. If not specified, no file will be saved."
        ),
    ] = None,
):
    """Cleans and deduplicates a text file using the Rust engine."""
    console.print(f"[bold cyan]Input file:[/] {filepath}")

    toxic_keywords = None
    if toxic_keywords_file:
        try:
            with open(toxic_keywords_file, "r", encoding="utf-8") as f:
                toxic_keywords = [line.strip() for line in f if line.strip()]
            console.print(
                f"[bold yellow]Toxic keyword filtering enabled:[/] {len(toxic_keywords)} keywords loaded"
            )
        except Exception as e:
            console.print(f"❌ [bold red]Failed to load toxic keywords file:[/] {e}")
            raise typer.Exit(code=1)

    try:
        cleaner = DataCleanser(min_length=min_length, toxic_keywords=toxic_keywords)
        with console.status("[bold green]Processing file with Rust engine...", spinner="dots"):
            cleaner.process_file(filepath)

        table = Table(
            title="Data Cleaning Results", show_header=True, header_style="bold magenta"
        )
        table.add_column("Item", style="dim", width=30)
        table.add_column("Value", justify="right")
        table.add_row("Input File", os.path.basename(filepath))
        table.add_row("Configured Min Sentence Length", str(min_length))
        table.add_row(
            "Number of Toxic Keywords Applied",
            str(len(toxic_keywords)) if toxic_keywords else "None",
        )
        table.add_row(
            "[bold green]Unique Sentences After Processing[/]",
            f"[bold green]{cleaner.count}[/]",
        )
        console.print(table)
        
        if output_file:
            with console.status(f"[bold yellow]Saving results to {output_file}...", spinner="dots"):
                cleaner.save_cleaned_to_file(output_file)
            console.print(
                f"[bold green]Task complete![/] {cleaner.count} unique sentences saved to {output_file}."
            )
        else:
            console.print(
                f"[bold green]Task complete![/] Found {cleaner.count} unique sentences. (Results not saved)"
            )

    except Exception as e:
        console.print(f"❌ [bold red]Error during file processing:[/] {e}")
        raise typer.Exit(code=1)

if TRAIN_AVAILABLE:
    @app.command()
    def train(
        model_name: Annotated[
            str, typer.Option(help="Name of the model to train (e.g., gpt2, distilgpt2)")
        ] = "gpt2",
        dataset_name: Annotated[
            Optional[str], typer.Option(help="Name of the dataset to use (e.g., wikitext). Cannot be used with --train-file.")
        ] = "wikitext",
        dataset_config: Annotated[
            Optional[str], typer.Option(help="Dataset configuration (e.g., wikitext-2-raw-v1)")
        ] = "wikitext-2-raw-v1",
        train_file: Annotated[
            Optional[str], typer.Option(help="Path to a local text file to use for training. Cannot be used with dataset_name.")
        ] = None,
        output_dir: Annotated[
            str, typer.Option(help="Output directory for the model")
        ] = "./output",
        num_train_epochs: Annotated[
            int, typer.Option(help="Number of training epochs")
        ] = 1,
        batch_size: Annotated[
            int, typer.Option(help="Batch size")
        ] = 4,
        learning_rate: Annotated[
            float, typer.Option(help="Learning rate")
        ] = 5e-5,
        use_telemetry: Annotated[
            bool, typer.Option(help="Enable experiment tracking and metric logging")
        ] = True,
        use_accelerate: Annotated[
            bool, typer.Option(help="Enable distributed training with Accelerate")
        ] = False,
    ):
        """Trains a text generation model using Hugging Face models."""
        if dataset_name and train_file:
            console.print("❌ [bold red]Error: Cannot specify both dataset_name and train_file.[/]")
            raise typer.Exit(code=1)
        if not dataset_name and not train_file:
            console.print("❌ [bold red]Error: Either dataset_name or train_file must be specified.[/]")
            raise typer.Exit(code=1)

        console.print(f"[bold blue]Starting model training for:[/] {model_name}")
        try:
            train_model(
                model_name=model_name,
                dataset_name=dataset_name if not train_file else None,
                dataset_config=dataset_config if not train_file else None,
                train_file=train_file,
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                use_telemetry=use_telemetry,
                use_accelerate=use_accelerate,
            )
            console.print("[bold green]Training complete![/]")
        except Exception as e:
            console.print(f"[bold red]Error during training:[/] {e}")
            raise typer.Exit(code=1)

if EVAL_AVAILABLE:
    @app.command()
    def evaluate(
        model_path: Annotated[
            str, typer.Argument(help="Path to the model to evaluate")
        ],
        tasks: Annotated[
            str, typer.Option(help="Tasks to evaluate (comma-separated)")
        ] = "hellaswag,winogrande,piqa",
        num_fewshot: Annotated[
            int, typer.Option(help="Number of few-shot samples")
        ] = 0,
        output_path: Annotated[
            str, typer.Option(help="Path to save results")
        ] = "evaluation_results.json",
    ):
        """Evaluates a model against standard benchmarks."""
        task_list = [t.strip() for t in tasks.split(",")]
        console.print(f"[bold blue]Starting model evaluation for:[/] {model_path}")
        console.print(f"[bold blue]Tasks:[/] {', '.join(task_list)}")
        try:
            evaluate_model(
                model_path=model_path,
                tasks=task_list,
                num_fewshot=num_fewshot,
                output_path=output_path,
            )
            console.print("[bold green]Evaluation complete![/]")
        except Exception as e:
            console.print(f"[bold red]Error during evaluation:[/] {e}")
            raise typer.Exit(code=1)

@app.command()
def upload(
    repository: Annotated[
        str, typer.Option(help="Repository to upload to (testpypi or pypi)")
    ] = "testpypi",
):
    """Builds and uploads the package to PyPI."""
    console.print(f"[bold blue]Starting PyPI upload to:[/] {repository}")
    _run_command(["python", "-m", "maturin", "build"], "Building package")

    upload_command = ["python", "-m", "twine", "upload"]
    if repository == "testpypi":
        upload_command.extend(["--repository", "testpypi"])
    upload_command.append("target/wheels/*")

    _run_command(upload_command, f"Uploading to {repository}")
    console.print("[bold green]Upload complete![/]")

def main():
    """Main function to run the CLI."""
    app()

if __name__ == "__main__":
    main()
