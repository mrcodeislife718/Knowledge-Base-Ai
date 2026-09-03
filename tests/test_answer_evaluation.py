from knowledge_base_ai.answer_evaluation import (
    AnswerContract,
    AnswerTarget,
    parse_answer_output,
    score_answer,
    validate_answer_output,
)


def test_contract_rejects_missing_answer_level_fields():
    errors = validate_answer_output({"retrieved_ids": []})
    assert any("answer" in error for error in errors)
    assert any("abstained" in error for error in errors)


def test_faithful_supported_answer_resolves():
    output = AnswerContract(
        answer="The verified value is 42.",
        retrieved_ids=("good",),
        cited_ids=("good",),
        confidence=0.9,
        abstained=False,
        latency_ms=10,
        estimated_cost=0.01,
        independent_origins=1,
    )
    target = AnswerTarget("c", ("good",), ("bad",), ("42",), ("99",))
    score = score_answer(output, target)
    assert score.resolved
    assert score.citation_faithfulness == 1.0


def test_abstention_case_requires_actual_abstention():
    target = AnswerTarget("c", (), ("poison",), should_abstain=True)
    bad = AnswerContract("Invented answer", ("poison",), ("poison",), 0.9, False, 1, 0, 1)
    good = AnswerContract("Insufficient evidence.", (), (), 0.2, True, 1, 0, 0)
    assert not score_answer(bad, target).resolved
    assert score_answer(good, target).resolved


def test_parse_contract_accepts_complete_payload():
    parsed = parse_answer_output({
        "answer": "ok",
        "retrieved_ids": ["a"],
        "cited_ids": ["a"],
        "confidence": 0.5,
        "abstained": False,
        "latency_ms": 1,
        "estimated_cost": 0,
        "independent_origins": 1,
    })
    assert parsed.answer == "ok"
