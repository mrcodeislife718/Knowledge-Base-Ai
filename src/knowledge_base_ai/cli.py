from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import requests
import typer
from rich.console import Console
from rich.table import Table

from .pipeline import DEFAULT_COLLECTION, DEFAULT_MODEL, ingest as run_ingest, latest_manifest
from .store import VectorStore
from .validation import validate_run

app = typer.Typer(no_args_is_help=True, help="Knowledge-Base AI: auditable document-to-vector pipeline.")
console = Console()


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
        manifest = run_ingest(
            source=source,
            workdir=workdir,
            title=title,
            author=author,
            force_ocr=force_ocr,
            model_name=model,
            collection_name=collection,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            verbose=verbose,
        )
    except Exception as exc:
        console.print(f"[bold red]Ingestion failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[bold green]Completed[/bold green] run {manifest.run_id}")
    console.print(
        f"pages={manifest.page_count} unique={manifest.unique_page_count} "
        f"duplicates={manifest.duplicate_page_count} chapters={manifest.chapter_count} chunks={manifest.chunk_count}"
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
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        console.rule(f"#{rank} {chunk_id} · distance={distance:.4f}")
        console.print(
            f"chapter={metadata.get('chapter')} pages={metadata.get('page_start')}-{metadata.get('page_end')} "
            f"method={metadata.get('extraction_methods')}"
        )
        console.print(f"source_sha256={metadata.get('source_sha256')}")
        console.print(document)


@app.command("validate")
def validate_command(
    workdir: Annotated[Path, typer.Option(help="Pipeline output directory")] = Path(".kbai"),
) -> None:
    """Run deterministic data, provenance, vector-count, and retrieval checks."""
    try:
        report = validate_run(workdir)
    except Exception as exc:
        console.print(f"[bold red]Validation failed to run:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    table = Table(title=f"Validation · run {report['run_id']}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, check in report["checks"].items():
        table.add_row(name, "PASS" if check["passed"] else "FAIL", check["detail"])
    console.print(table)
    console.print(f"score={report['score']:.0%}")
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
    """Resolve the Library of Congress public-domain Alice record and list PDF candidates."""
    record_url = "https://www.loc.gov/item/02020394/?fo=json"
    try:
        response = requests.get(record_url, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        console.print(f"[bold red]Could not fetch Library of Congress record:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    candidates: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str) and ".pdf" in value.lower() and value.startswith("http"):
            candidates.append(value)

    walk(payload.get("resources", payload))
    console.print("[bold]Alice's Adventures in Wonderland (1895) — Library of Congress[/bold]")
    console.print("Record: https://www.loc.gov/item/02020394/")
    if candidates:
        for url in sorted(set(candidates)):
            console.print(url)
    else:
        console.print("No direct PDF URL was exposed in the JSON response; use the record's Download menu.")


if __name__ == "__main__":
    app()
