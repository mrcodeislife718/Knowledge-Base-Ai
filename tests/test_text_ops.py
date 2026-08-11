from knowledge_base_ai.models import PageRecord
from knowledge_base_ai.text_ops import assign_chapters, clean_text, deduplicate_pages, semantic_chunks


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


def test_chapter_detection_and_chunking_preserve_boundaries():
    pages = [
        page(1, "CHAPTER I\n\nDown the Rabbit-Hole\n\nAlice was beginning to get very tired."),
        page(2, "She saw a White Rabbit with pink eyes.\n\nIt ran close by her."),
        page(3, "CHAPTER II\n\nThe Pool of Tears\n\nAlice grew and became confused."),
    ]
    assign_chapters(pages)
    chunks = semantic_chunks(pages, target_chars=120, overlap_chars=20)
    assert pages[0].chapter == "Chapter I"
    assert pages[2].chapter == "Chapter II"
    assert {chunk.chapter for chunk in chunks} == {"Chapter I", "Chapter II"}
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
