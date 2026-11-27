import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    duplicate_rate,
)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"A", "C"}

    assert precision_at_k(retrieved, relevant, 4) == 0.5


def test_recall_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C", "D"}

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_reciprocal_rank_first_result():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C"}

    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_reciprocal_rank_second_result():
    retrieved = ["A", "B", "C"]
    relevant = {"B", "C"}

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_reciprocal_rank_no_relevant_result():
    retrieved = ["A", "B", "C"]
    relevant = {"D"}

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_duplicate_rate():
    results = [
        {"document": "Annual leave is 30 days."},
        {"document": "Password recovery through IT."},
        {"document": "ANNUAL LEAVE IS 30 DAYS."},
    ]

    assert duplicate_rate(results) == pytest.approx(1 / 3)


def test_duplicate_rate_empty_results():
    assert duplicate_rate([]) == 0.0


def test_precision_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, 0)


def test_recall_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, 0)


def test_ndcg_at_k_ideal_ranking():
    from evaluation.metrics import ndcg_at_k

    grades = {"A": 3, "B": 1, "C": 2}

    assert ndcg_at_k(["A", "C", "B"], grades, 3) == 1.0


def test_ndcg_at_k_poor_ranking():
    from evaluation.metrics import ndcg_at_k

    grades = {"A": 3, "B": 1, "C": 2}

    score = ndcg_at_k(["B", "C", "A"], grades, 3)

    assert 0 < score < 1


def test_ndcg_at_k_no_relevant_results():
    from evaluation.metrics import ndcg_at_k

    grades = {"A": 0, "B": 0}

    assert ndcg_at_k(["A", "B"], grades, 2) == 0.0


def test_ndcg_at_k_empty_results():
    from evaluation.metrics import ndcg_at_k

    grades = {"A": 3, "B": 2}

    assert ndcg_at_k([], grades, 2) == 0.0


def test_ndcg_at_k_invalid_k():
    import pytest
    from evaluation.metrics import ndcg_at_k

    with pytest.raises(ValueError):
        ndcg_at_k(["A"], {"A": 3}, 0)


def test_ndcg_at_k_invalid_relevance_grade():
    import pytest
    from evaluation.metrics import ndcg_at_k

    with pytest.raises(ValueError):
        ndcg_at_k(["A"], {"A": -1}, 1)
