# LMDK (Language Model Development Kit)

🚀 **Industry-standard development kit for building custom Language Models**

LMDK eliminates complexity by providing an integrated solution for data pipeline optimization, distributed training orchestration, and experiment reproducibility.

## Features

- **High Performance**: 10x speedup in data processing with Rust core
- **Ease of Use**: Simple CLI commands for complex workflows
- **Scientific Reproducibility**: Automatic experiment tracking with MLflow
- **Distributed Training**: Multi-GPU support with Hugging Face Accelerate
- **Model Optimization**: Built-in quantization for efficient inference

## Quick Start: A Full Workflow Example

This section walks you through a complete workflow, from data preparation to model deployment.

### Installation

```bash
pip install lmdk
```

### Full Workflow

1.  **Prepare your data**: Clean and deduplicate your text data, filtering out unwanted content.

    Using `sample_data.txt` and `toxic.txt` from this project:
    ```bash
    lmdk prep sample_data.txt --min-length 20 --toxic-keywords-file toxic.txt --output-file prepared_data.txt
    ```
    This command processes `sample_data.txt`, keeps lines longer than 20 characters, filters out words found in `toxic.txt`, and saves the unique, cleaned lines to `prepared_data.txt`.

2.  **Train a model**: Fine-tune a Hugging Face causal language model using your prepared data.

    ```bash
    lmdk train --model-name gpt2 --train-file prepared_data.txt --output-dir ./my_model --num-train-epochs 1 --batch-size 2
    ```
    This command fine-tunes a `gpt2` model for 1 epoch with a batch size of 2, using `prepared_data.txt` as the training data. The trained model will be saved to `./my_model`.

3.  **Evaluate performance**: Benchmark the trained model on standard tasks.

    ```bash
    lmdk evaluate ./my_model --tasks hellaswag,winogrande --output-path evaluation_results.json
    ```
    This evaluates your model on the `hellaswag` and `winogrande` benchmarks, saving the results to `evaluation_results.json`.

4.  **Optimize for deployment**: Quantize the model for efficient inference.

    ```bash
    lmdk quantize ./my_model ./optimized_model --bits 8
    ```
    This quantizes your trained model (`./my_model`) to an 8-bit version, saving the optimized model to `./optimized_model`.

## Architecture

LMDK uses a hybrid Python-Rust architecture:

-   **Python Control Layer**: CLI, training orchestration, experiment management
-   **Rust High-Performance Engine**: Data processing, tokenization, optimization
-   **ML Framework Integration**: PyTorch, Hugging Face Transformers

## CLI Commands

-   `lmdk prep <filepath> --min-length <int> --toxic-keywords-file <path> --output-file <path>`: Data preprocessing and cleaning.
    -   `filepath`: Path to the input text file.
    -   `--min-length`: Minimum length for a valid sentence.
    -   `--toxic-keywords-file`: Path to a file containing toxic keywords (one per line) for filtering.
    -   `--output-file`: Path to save the cleaned and deduplicated data. If not provided, the cleaned data is not saved.
-   `lmdk train --model-name <str> (--dataset-name <str> | --train-file <path>) [--dataset-config <str>] --output-dir <path> [--num-train-epochs <int>] [--batch-size <int>] [--learning-rate <float>]`: Model training with distributed support.
    -   `--model-name`: Name of the Hugging Face model to train (e.g., `gpt2`).
    -   `--dataset-name`: Name of a Hugging Face dataset to use for training (cannot be used with `--train-file`).
    -   `--train-file`: Path to a local text file to use for training (cannot be used with `--dataset-name`).
    -   `--dataset-config`: Configuration for the Hugging Face dataset.
    -   `--output-dir`: Directory to save the trained model.
    -   `--num-train-epochs`: Number of training epochs.
    -   `--batch-size`: Training batch size.
    -   `--learning-rate`: Learning rate for training.
-   `lmdk evaluate <model_path> --tasks <str> [--num-fewshot <int>] [--output-path <path>]`: Benchmark evaluation.
    -   `model_path`: Path to the trained model.
    -   `--tasks`: Comma-separated list of evaluation tasks (e.g., `hellaswag,winogrande`).
    -   `--num-fewshot`: Number of few-shot samples to use.
    -   `--output-path`: Path to save evaluation results (JSON format).
-   `lmdk quantize <model_path> <output_path> --bits <int>`: Model quantization.
    -   `model_path`: Path to the model to quantize.
    -   `output_path`: Path to save the quantized model.
    -   `--bits`: Number of bits for quantization (4 or 8).
-   `lmdk upload --repository <str>`: PyPI package upload.
    -   `--repository`: Repository to upload to (`testpypi` or `pypi`).

## Requirements

-   **Supported OS**: Windows, Linux
-   Python 3.9+
-   Rust 1.70+
-   PyTorch
-   Hugging Face Transformers

## Development

To set up a development environment, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/lmdk.git
    cd lmdk
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```
4.  **Build the Rust extension:**
    ```bash
    maturin develop
    ```
5.  **Run tests:**
    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.