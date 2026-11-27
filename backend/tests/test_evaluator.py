import pytest

from evaluation.evaluator import evaluate_retrieval


class FakeSearchService:
    """Deterministic search service for evaluator tests."""

    def search(self, query: str, top_k: int = 5):
        results = {
            "Question 1": [
                {"id": "A", "document": "Annual leave policy"},
                {"id": "B", "document": "Working hours policy"},
                {"id": "C", "document": "Password recovery"},
            ],
            "Question 2": [
                {"id": "D", "document": "Password reset instructions"},
                {"id": "E", "document": "IT support instructions"},
            ],
        }

        return results.get(query, [])[:top_k]


@pytest.fixture
def fake_dataset(monkeypatch):
    dataset = [
        {
            "id": "Q001",
            "question": "Question 1",
            "relevant_chunk_ids": ["A", "B"],
            "relevance_grades": {"A": 3, "B": 2},
            "category": "hr_policy",
        },
        {
            "id": "Q002",
            "question": "Question 2",
            "relevant_chunk_ids": ["D"],
            "relevance_grades": {"D": 3},
            "category": "it_support",
        },
    ]

    monkeypatch.setattr(
        "evaluation.evaluator.load_dataset",
        lambda: dataset,
    )

    return dataset


def test_evaluate_retrieval_returns_expected_metrics(fake_dataset):
    report = evaluate_retrieval(
        search_service=FakeSearchService(),
        top_k=3,
    )

    assert report["evaluation_questions"] == 2
    assert report["top_k"] == 3

    metrics = report["metrics"]

    assert metrics["mean_precision_at_k"] == 0.5833
    assert metrics["mean_recall_at_k"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0
    assert metrics["mean_ndcg_at_k"] == 1.0
    assert metrics["mean_duplicate_rate"] == 0.0


def test_evaluate_retrieval_returns_question_results(fake_dataset):
    report = evaluate_retrieval(
        search_service=FakeSearchService(),
        top_k=3,
    )

    assert len(report["results"]) == 2

    first_result = report["results"][0]

    assert first_result["question_id"] == "Q001"
    assert first_result["retrieved_count"] == 3
    assert first_result["relevant_count"] == 2
    assert first_result["ndcg_at_k"] == 1.0


def test_evaluate_retrieval_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        evaluate_retrieval(
            search_service=FakeSearchService(),
            top_k=0,
        )


def test_evaluate_retrieval_rejects_empty_dataset(monkeypatch):
    monkeypatch.setattr(
        "evaluation.evaluator.load_dataset",
        lambda: [],
    )

    with pytest.raises(ValueError, match="dataset is empty"):
        evaluate_retrieval(
            search_service=FakeSearchService(),
            top_k=3,
        )
def test_load_dataset_v1():
    from evaluation.evaluator import (
        load_dataset,
        DEFAULT_DATASET_PATH,
    )

    dataset = load_dataset()

    assert DEFAULT_DATASET_PATH.exists()
    assert len(dataset) == 5


def test_load_dataset_v2():
    from evaluation.evaluator import (
        load_dataset,
        V2_DATASET_PATH,
    )

    dataset = load_dataset(V2_DATASET_PATH)

    assert V2_DATASET_PATH.exists()
    assert len(dataset) == 50


def test_evaluate_retrieval_records_dataset_name(fake_dataset):
    report = evaluate_retrieval(
        search_service=FakeSearchService(),
        top_k=3,
    )

    assert report["dataset"] == "retrieval_questions.json"