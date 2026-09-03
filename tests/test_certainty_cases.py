from knowledge_base_ai.certainty_cases import certainty_heldout_cases


def test_certainty_suite_covers_all_required_categories_and_size():
    cases = certainty_heldout_cases(seed=123, per_category=40)
    categories = {case.category for case in cases}
    assert len(cases) >= 300
    assert {
        "lookup",
        "contradiction",
        "multi-hop",
        "stale-knowledge",
        "source-independence",
        "adversarial",
        "citation-faithfulness",
        "abstention",
        "global-synthesis",
    }.issubset(categories)
    assert len({case.case_id for case in cases}) == len(cases)
