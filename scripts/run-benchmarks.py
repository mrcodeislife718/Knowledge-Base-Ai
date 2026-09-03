from pathlib import Path

from knowledge_base_ai.benchmarking import run_suite, write_report


def main() -> None:
    result = run_suite()
    json_path, md_path = write_report(Path("benchmark-results"), result)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
