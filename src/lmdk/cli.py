# lmdk/cli.py
import sys
import os

# ModuleNotFoundError 수정을 위해 프로젝트 루트를 sys.path에 추가
# src layout으로 변경됨에 따라 이 코드는 불필요할 수 있으나, 만약을 위해 유지합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

# rust_core 라이브러리 임포트
from rust_core import DataCleanser

app = typer.Typer(
    name="lmdk",
    help="🚀 Language Model Development Kit - 데이터 파이프라인부터 모델 학습까지",
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
    📝 텍스트 파일을 정제하고 중복을 제거합니다.
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


if __name__ == "__main__":
    app()
