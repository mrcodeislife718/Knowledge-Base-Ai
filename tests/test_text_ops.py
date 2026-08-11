from knowledge_base_ai.models import PageRecord
from knowledge_base_ai.text_ops import (
    assign_chapters,
    clean_text,
    deduplicate_pages,
    semantic_chunks,
)


def page(number: int, text: str) -> PageRecord:
    return PageRecord(
        page_number=number,
        text=text,
        raw_text=text,
        extraction_method="native-text",
        source_path="book.pdf",
        source_sha256="sourcehash",
        page_sha256="",
    )


def test_clean_text_repairs_linebreak_hyphenation():
    assert clean_text("won-\nderland\n\n\nAlice") == "wonderland\n\nAlice"


def test_deduplicate_pages_tracks_exact_duplicate():
    pages = [page(1, "Same text"), page(2, " Same   text "), page(3, "Different")]
    unique, duplicates = deduplicate_pages(pages)
    assert duplicates == 1
    assert [item.page_number for item in unique] == [1, 3]
    assert pages[1].duplicate_of == 1


def test_chapter_detection_ignores_running_headers():
    pages = [
        page(1, "ALICE'S ADVENTURES IN WONDERLAND\n\nFront matter text goes here."),
        page(2, "CHAPTER I\n\nDown the Rabbit-Hole\n\nAlice was beginning to get very tired."),
    ]
    assign_chapters(pages)
    assert pages[0].chapter == "Front Matter"
    assert pages[1].chapter == "Chapter I"


def test_chapter_detection_and_chunking_preserve_boundaries():
    pages = [
        page(
            1,
            "CHAPTER I\n\nDown the Rabbit-Hole\n\n"
            + "Alice was beginning to get very tired of sitting by her sister. " * 8,
        ),
        page(
            2,
            "She saw a White Rabbit with pink eyes. "
            + "It ran close by her and disappeared down the rabbit hole. " * 8,
        ),
        page(
            3,
            "CHAPTER II\n\nThe Pool of Tears\n\n"
            + "Alice grew and became confused about what was happening. " * 8,
        ),
    ]
    assign_chapters(pages)
    chunks = semantic_chunks(pages, target_chars=300, overlap_chars=40)
    assert pages[0].chapter == "Chapter I"
    assert pages[2].chapter == "Chapter II"
    assert {chunk.chapter for chunk in chunks} == {"Chapter I", "Chapter II"}
    assert all(80 <= len(chunk.text) <= 500 for chunk in chunks)
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
