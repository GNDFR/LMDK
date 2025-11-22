# lmdk/cli.py
import sys
import os

# ModuleNotFoundError 수정을 위해 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

# rust_core 라이브러리 임포트
from rust_core import DataCleanser
import subprocess

try:
    from rust_core import ModelQuantizer
    QUANTIZER_AVAILABLE = True
except ImportError:
    QUANTIZER_AVAILABLE = False

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
    help="Language Model Development Kit - 데이터 파이프라인부터 모델 학습까지",
    add_completion=False,
)
console = Console()


@app.command()
def prep(
    filepath: Annotated[
        str,
        typer.Argument(
            help="처리할 텍스트 파일의 경로",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    min_length: Annotated[
        int, typer.Option(help="유효한 문장으로 간주할 최소 길이")
    ] = 20,
    toxic_keywords_file: Annotated[
        str,
        typer.Option(
            help="필터링할 유해 단어 목록 파일 (한 줄에 한 단어)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
):
    """
    텍스트 파일을 정제하고 중복을 제거합니다.
    """
    console.print(f"📁 [bold cyan]입력 파일:[/] {filepath}")

    toxic_keywords = None
    if toxic_keywords_file:
        try:
            with open(toxic_keywords_file, "r", encoding="utf-8") as f:
                toxic_keywords = [line.strip() for line in f if line.strip()]
            console.print(
                f"🚫 [bold yellow]유해 단어 필터링 활성화:[/] {len(toxic_keywords)}개 단어 로드됨"
            )
        except Exception as e:
            console.print(f"❌ [bold red]유해 단어 파일 로드 실패:[/] {e}")
            raise typer.Exit(code=1)

    try:
        # DataCleanser 초기화
        cleaner = DataCleanser(min_length=min_length, toxic_keywords=toxic_keywords)

        # 파일 처리
        with console.status(
            "[bold green]🦀 Rust 엔진으로 파일 처리 중...", spinner="dots"
        ):
            processed_count = cleaner.process_file(filepath)

        # 결과 출력
        table = Table(
            title="📊 데이터 정제 결과", show_header=True, header_style="bold magenta"
        )
        table.add_column("항목", style="dim", width=30)
        table.add_column("값", justify="right")

        table.add_row("입력 파일", os.path.basename(filepath))
        table.add_row("설정된 최소 문장 길이", str(min_length))
        table.add_row(
            "적용된 유해 단어 수",
            str(len(toxic_keywords)) if toxic_keywords else "기본값",
        )
        table.add_row(
            "[bold green]처리 후 고유 문장 수[/]",
            f"[bold green]{cleaner.count}[/]",
        )

        console.print(table)
        console.print(
            f"✅ [bold green]작업 완료![/] 최종적으로 {cleaner.count}개의 고유한 문장이 저장되었습니다."
        )

    except Exception as e:
        console.print(f"❌ [bold red]파일 처리 중 오류 발생:[/] {e}")
        raise typer.Exit(code=1)


if TRAIN_AVAILABLE:
    @app.command()
    def train(
        model_name: Annotated[
            str, typer.Option(help="훈련할 모델 이름 (예: gpt2, distilgpt2)")
        ] = "gpt2",
        dataset_name: Annotated[
            str, typer.Option(help="사용할 데이터셋 이름")
        ] = "wikitext",
        dataset_config: Annotated[
            str, typer.Option(help="데이터셋 설정")
        ] = "wikitext-2-raw-v1",
        output_dir: Annotated[
            str, typer.Option(help="모델 출력 디렉토리")
        ] = "./output",
        num_train_epochs: Annotated[
            int, typer.Option(help="훈련 에폭 수")
        ] = 1,
        batch_size: Annotated[
            int, typer.Option(help="배치 크기")
        ] = 4,
        learning_rate: Annotated[
            float, typer.Option(help="학습률")
        ] = 5e-5,
        use_telemetry: Annotated[
            bool, typer.Option(help="실험 추적 및 메트릭 로깅 활성화")
        ] = True,
        use_accelerate: Annotated[
            bool, typer.Option(help="Accelerate를 사용한 분산 훈련 활성화")
        ] = False,
    ):
        """
        Hugging Face 모델을 사용하여 텍스트 생성 모델을 훈련합니다.
        """
        console.print(f"[bold blue]모델 훈련 시작:[/] {model_name}")

        try:
            train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                use_telemetry=use_telemetry,
                use_accelerate=use_accelerate,
            )
            console.print("[bold green]훈련 완료![/]")
        except Exception as e:
            console.print(f"[bold red]훈련 중 오류 발생:[/] {e}")
            raise typer.Exit(code=1)


if EVAL_AVAILABLE:
    @app.command()
    def evaluate(
        model_path: Annotated[
            str, typer.Argument(help="평가할 모델 경로")
        ],
        tasks: Annotated[
            str, typer.Option(help="평가할 태스크들 (쉼표로 구분)")
        ] = "hellaswag,winogrande,piqa",
        num_fewshot: Annotated[
            int, typer.Option(help="Few-shot 샘플 수")
        ] = 0,
        output_path: Annotated[
            str, typer.Option(help="결과 저장 경로")
        ] = "evaluation_results.json",
    ):
        """
        표준 벤치마크로 모델을 평가합니다.
        """
        task_list = [t.strip() for t in tasks.split(",")]
        console.print(f"[bold blue]모델 평가 시작:[/] {model_path}")
        console.print(f"[bold blue]태스크들:[/] {', '.join(task_list)}")

        try:
            results = evaluate_model(
                model_path=model_path,
                tasks=task_list,
                num_fewshot=num_fewshot,
                output_path=output_path,
            )
            console.print("[bold green]평가 완료![/]")
        except Exception as e:
            console.print(f"[bold red]평가 중 오류 발생:[/] {e}")
            raise typer.Exit(code=1)


@app.command()
def upload(
    repository: Annotated[
        str, typer.Option(help="업로드할 저장소 (testpypi 또는 pypi)")
    ] = "testpypi",
):
    """
    PyPI에 패키지를 업로드합니다.
    """
    console.print(f"[bold blue]PyPI 업로드 시작:[/] {repository}")

    try:
        # Build the package
        console.print("[dim]패키지 빌드 중...[/]")
        subprocess.run(["python", "-m", "maturin", "build"], check=True)

        # Upload to PyPI
        console.print(f"[dim]{repository}에 업로드 중...[/]")
        if repository == "testpypi":
            subprocess.run(["python", "-m", "twine", "upload", "--repository", "testpypi", "target/wheels/*"], check=True)
        else:
            subprocess.run(["python", "-m", "twine", "upload", "target/wheels/*"], check=True)

        console.print("[bold green]업로드 완료![/]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]업로드 실패:[/] {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError:
        console.print("[bold red]twine이 설치되지 않았습니다. pip install twine을 실행하세요.[/]")
        raise typer.Exit(code=1)


if QUANTIZER_AVAILABLE:
    @app.command()
    def quantize(
        model_path: Annotated[
            str, typer.Argument(help="양자화할 모델 경로")
        ],
        output_path: Annotated[
            str, typer.Argument(help="출력 경로")
        ],
        bits: Annotated[
            int, typer.Option(help="양자화 비트 수 (4 또는 8)")
        ] = 8,
    ):
        """
        모델을 양자화합니다 (4-bit 또는 8-bit).
        """
        console.print(f"[bold blue]모델 양자화 시작:[/] {model_path} -> {output_path} ({bits}-bit)")

        try:
            quantizer = ModelQuantizer()
            if bits == 8:
                quantizer.quantize_8bit(model_path, output_path)
            elif bits == 4:
                quantizer.quantize_4bit(model_path, output_path)
            else:
                raise ValueError("bits must be 4 or 8")

            console.print("[bold green]양자화 완료![/]")
        except Exception as e:
            console.print(f"[bold red]양자화 중 오류 발생:[/] {e}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
