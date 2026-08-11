from knowledge_base_ai.quality import classify_chunk, classify_page, score_text_quality


def test_clean_text_scores_as_readable():
    text = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, "
        "and of having nothing to do."
    )
    score, flags = score_text_quality(text)
    assert score >= 0.72
    assert "low_quality" not in flags


def test_empty_text_is_flagged():
    score, flags = score_text_quality("   \n")
    assert score == 0.0
    assert flags == ["empty_text"]


def test_page_classification_respects_front_matter_and_chapter():
    assert classify_page("TABLE OF CONTENTS", "Front Matter") == "table-of-contents"
    assert classify_page("CHAPTER I\nDown the Rabbit-Hole", "Chapter I") == "chapter-opening"


def test_dialogue_heavy_chunk_is_labeled():
    text = '"Hello," said Alice. "Where?" asked the Rabbit. "Here," she replied. "Now!"'
    assert classify_chunk(text) == "dialogue-heavy-narrative"
