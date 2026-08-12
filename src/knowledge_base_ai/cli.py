from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import requests
import typer
from rich.console import Console
from rich.table import Table

from .pipeline import DEFAULT_COLLECTION, DEFAULT_MODEL, latest_manifest
from .pipeline import ingest as run_ingest
from .store import VectorStore
from .validation import validate_run

app = typer.Typer(no_args_is_help=True, help="Knowledge-Base AI: auditable document-to-vector pipeline.")
console = Console()
DEMO_PDF_URL = "https://archive.org/download/alicesadventures00carr_17/alicesadventures00carr_17.pdf"


def _print_validation(report: dict) -> None:
    table = Table(title=f"Validation · run {report['run_id']}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, check in report["checks"].items():
        table.add_row(name, "PASS" if check["passed"] else "FAIL", check["detail"])
    console.print(table)
    console.print(f"score={report['score']:.0%}")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as fh:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    fh.write(block)
    partial.replace(destination)


@app.command("ingest")
def ingest_command(
    source: Annotated[Path, typer.Argument(help="PDF file or directory of page images")],
    title: Annotated[str | None, typer.Option(help="Override extracted title metadata")] = None,
    author: Annotated[str | None, typer.Option(help="Override extracted author metadata")] = None,
    force_ocr: Annotated[bool, typer.Option(help="OCR every page instead of preferring native text")] = False,
    model: Annotated[str, typer.Option(help="SentenceTransformer model name")] = DEFAULT_MODEL,
    collection: Annotated[str, typer.Option(help="Chroma collection name")] = DEFAULT_COLLECTION,
    workdir: Annotated[Path, typer.Option(help="Pipeline output directory")] = Path(".kbai"),
    target_chars: Annotated[int, typer.Option(help="Approximate semantic chunk target size")] = 1200,
    overlap_chars: Annotated[int, typer.Option(help="Approximate chunk overlap size")] = 180,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run extraction/OCR through Chroma ingestion."""
    try:
        manifest = run_ingest(source, workdir, title, author, force_ocr, model, collection, target_chars, overlap_chars, verbose)
    except Exception as exc:
        console.print(f"[bold red]Ingestion failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[bold green]Completed[/bold green] run {manifest.run_id}")
    console.print(
        f"pages={manifest.page_count} unique={manifest.unique_page_count} duplicates={manifest.duplicate_page_count} "
        f"low_quality={manifest.low_quality_page_count} chapters={manifest.chapter_count} chunks={manifest.chunk_count}"
    )


@app.command("query")
def query_command(
    text: Annotated[str, typer.Argument(help="Natural-language retrieval query")],
    top_k: Annotated[int, typer.Option(help="Number of results")] = 5,
    workdir: Annotated[Path, typer.Option(help="Pipeline output directory")] = Path(".kbai"),
) -> None:
    """Retrieve semantically similar chunks with provenance."""
    try:
        manifest = latest_manifest(workdir)
        store = VectorStore(workdir / "chroma", manifest["collection_name"], manifest["embedding_model"])
        results = store.query(text, top_k=top_k)
    except Exception as exc:
        console.print(f"[bold red]Query failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    for rank, (chunk_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        console.rule(f"#{rank} {chunk_id} · distance={distance:.4f}")
        console.print(
            f"chapter={metadata.get('chapter')} pages={metadata.get('page_start')}-{metadata.get('page_end')} "
            f"label={metadata.get('semantic_label')} method={metadata.get('extraction_methods')}"
        )
        console.print(f"source_sha256={metadata.get('source_sha256')}")
        console.print(document)


@app.command("validate")
def validate_command(
    workdir: Annotated[Path, typer.Option(help="Pipeline output directory")] = Path(".kbai"),
) -> None:
    """Run deterministic data, provenance, vector-count, embedding, and retrieval checks."""
    try:
        report = validate_run(workdir)
    except Exception as exc:
        console.print(f"[bold red]Validation failed to run:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    _print_validation(report)
    if not report["passed"]:
        raise typer.Exit(2)


@app.command("manifest")
def manifest_command(
    workdir: Annotated[Path, typer.Option(help="Pipeline output directory")] = Path(".kbai"),
) -> None:
    """Print the latest provenance/run manifest."""
    try:
        console.print_json(json.dumps(latest_manifest(workdir)))
    except Exception as exc:
        console.print(f"[bold red]Manifest unavailable:[/bold red] {exc}")
        raise typer.Exit(1) from exc


@app.command("demo-source")
def demo_source() -> None:
    """Print the fixed public-domain scan used by the reproducible demo."""
    console.print("[bold]Alice's Adventures in Wonderland (1895)[/bold]")
    console.print("Library of Congress record: https://www.loc.gov/item/02020394/")
    console.print(f"Internet Archive PDF mirror: {DEMO_PDF_URL}")


@app.command("demo")
def demo(
    workdir: Annotated[Path, typer.Option(help="Isolated demo output directory")] = Path(".kbai-demo"),
    force_ocr: Annotated[bool, typer.Option(help="Force Tesseract OCR on all pages (slower)")] = False,
    fresh: Annotated[bool, typer.Option(help="Delete the prior demo output first")] = True,
) -> None:
    """Download a public-domain scan, ingest it, validate it, and run a retrieval query."""
    if fresh and workdir.exists():
        shutil.rmtree(workdir)
    source = workdir / "source" / "alice-1895.pdf"
    try:
        if not source.exists():
            console.print("[bold]1/4 Downloading public-domain 1895 scan...[/bold]")
            _download(DEMO_PDF_URL, source)
        else:
            console.print("[bold]1/4 Using cached demo scan.[/bold]")

        console.print("[bold]2/4 Running OCR-aware ingestion and vectorization...[/bold]")
        manifest = run_ingest(
            source=source,
            workdir=workdir,
            title="Alice's Adventures in Wonderland",
            author="Lewis Carroll",
            force_ocr=force_ocr,
            collection_name="knowledge-base-ai-demo",
        )

        console.print("[bold]3/4 Running quality validation...[/bold]")
        report = validate_run(workdir)
        _print_validation(report)
        if not report["passed"]:
            raise RuntimeError("Demo validation did not pass all quality gates.")

        console.print("[bold]4/4 Running retrieval proof...[/bold]")
        store = VectorStore(workdir / "chroma", manifest.collection_name, manifest.embedding_model)
        results = store.query("Why does Alice follow the White Rabbit?", top_k=3)
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        for rank, (chunk_id, document, metadata) in enumerate(zip(ids, docs, metas), start=1):
            console.rule(f"Demo result #{rank} · {chunk_id}")
            console.print(f"chapter={metadata.get('chapter')} pages={metadata.get('page_start')}-{metadata.get('page_end')}")
            console.print(document[:900])
        console.print(f"[bold green]DEMO PASSED[/bold green] run={manifest.run_id}")
        console.print(f"inventory={manifest.inventory_path}")
        console.print(f"knowledge_tree={manifest.knowledge_tree_path}")
    except Exception as exc:
        console.print(f"[bold red]DEMO FAILED:[/bold red] {exc}")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
