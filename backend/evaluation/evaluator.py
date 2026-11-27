import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.hybrid_search import HybridSearch

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    duplicate_rate,
    ndcg_at_k,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEFAULT_RRF_K = 40


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

DEFAULT_DATASET_PATH = (
    DATASET_DIR / "retrieval_questions.json"
)

V2_DATASET_PATH = (
    DATASET_DIR / "retrieval_questions_v2.json"
)

V3_DATASET_PATH = (
    DATASET_DIR / "retrieval_questions_v3.json"
)

REPORTS_DIR = BASE_DIR / "reports"


# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

def load_dataset(
    dataset_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load evaluation questions from a JSON dataset."""

    path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "Evaluation dataset must be a JSON list."
        )

    return dataset


# --------------------------------------------------
# RETRIEVAL METRICS
# --------------------------------------------------

def calculate_metrics(
    results: list[dict[str, Any]],
    relevant_ids: set[str],
    relevance_grades: dict[str, int],
    top_k: int,
) -> dict[str, float]:
    """Calculate retrieval metrics for a single question."""

    retrieved_ids = [
        result["id"]
        for result in results
    ]

    return {
        "precision_at_k": precision_at_k(
            retrieved_ids,
            relevant_ids,
            top_k,
        ),

        "recall_at_k": recall_at_k(
            retrieved_ids,
            relevant_ids,
            top_k,
        ),

        "reciprocal_rank": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),

        "ndcg_at_k": ndcg_at_k(
            retrieved_ids,
            relevance_grades,
            top_k,
        ),

        "duplicate_rate": duplicate_rate(results),
    }


# --------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------

def save_evaluation_report(
    report: dict[str, Any],
) -> Path:
    """Save an evaluation report using a unique UTC timestamp."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    report_path = REPORTS_DIR / f"evaluation_{timestamp}.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    return report_path


# --------------------------------------------------
# RETRIEVAL EVALUATION
# --------------------------------------------------

def evaluate_retrieval(
    search_service: Any = None,
    top_k: int = 5,
    dataset_path: str | Path | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, Any]:
    """
    Evaluate retrieval against a labelled dataset.

    Calculates:
    - Precision@K
    - Recall@K
    - Mean Reciprocal Rank
    - NDCG@K
    - Duplicate Rate
    - Category-level metrics
    - Query latency

    Args:
        search_service: Search service to evaluate.
        top_k: Number of results to retrieve.
        dataset_path: Optional evaluation dataset path.
        rrf_k: RRF smoothing constant used when creating
               the default HybridSearch service.

    Returns:
        Complete evaluation report.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if rrf_k <= 0:
        raise ValueError(
            "rrf_k must be greater than zero."
        )

    # Create a configured hybrid search service
    # when no custom service is provided.
    if search_service is None:
        search_service = HybridSearch(rrf_k=rrf_k)

    # Preserve compatibility with tests that replace
    # load_dataset() with a zero-argument function.
    if dataset_path is None:
        dataset = load_dataset()
        selected_dataset = DEFAULT_DATASET_PATH
    else:
        dataset = load_dataset(dataset_path)
        selected_dataset = Path(dataset_path)

    if not dataset:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    question_results = []

    category_results = {}

    total_metrics = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "reciprocal_rank": 0.0,
        "ndcg_at_k": 0.0,
        "duplicate_rate": 0.0,
    }

    latencies = []

    # --------------------------------------------------
    # EVALUATE EACH QUESTION
    # --------------------------------------------------

    for item in dataset:

        question = item["question"]

        category = item.get(
            "category",
            "uncategorized",
        )

        relevant_ids = set(
            item["relevant_chunk_ids"]
        )

        relevance_grades = item["relevance_grades"]

        start_time = time.perf_counter()

        results = search_service.search(
            query=question,
            top_k=top_k,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        latencies.append(latency_ms)

        metrics = calculate_metrics(
            results=results,
            relevant_ids=relevant_ids,
            relevance_grades=relevance_grades,
            top_k=top_k,
        )

        retrieved_ids = [
            result["id"]
            for result in results
        ]

        # QUESTION-LEVEL RESULTS

        question_results.append({
            "question_id": item["id"],
            "question": question,
            "category": category,

            "retrieved_count": len(results),
            "relevant_count": len(relevant_ids),

            **{
                metric: round(value, 4)
                for metric, value in metrics.items()
            },

            "latency_ms": round(
                latency_ms,
                2,
            ),

            "retrieved_chunk_ids": retrieved_ids,
        })

        # OVERALL AGGREGATION

        for metric_name, value in metrics.items():
            total_metrics[metric_name] += value

        # CATEGORY-LEVEL AGGREGATION

        if category not in category_results:

            category_results[category] = {
                metric: 0.0
                for metric in total_metrics
            }

            category_results[category]["question_count"] = 0

        for metric_name, value in metrics.items():
            category_results[category][metric_name] += value

        category_results[category]["question_count"] += 1

    # --------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------

    question_count = len(dataset)

    mean_metrics = {
        f"mean_{metric}": round(
            value / question_count,
            4,
        )
        for metric, value in total_metrics.items()
    }

    # --------------------------------------------------
    # CATEGORY SUMMARY
    # --------------------------------------------------

    category_summary = {}

    for category, values in category_results.items():

        count = values["question_count"]

        category_summary[category] = {
            "question_count": count,

            **{
                f"mean_{metric}": round(
                    values[metric] / count,
                    4,
                )
                for metric in total_metrics
            },
        }

    # --------------------------------------------------
    # LATENCY SUMMARY
    # --------------------------------------------------

    first_query_latency = latencies[0]

    subsequent_latencies = latencies[1:]

    mean_subsequent_latency = (
        sum(subsequent_latencies) / len(subsequent_latencies)
        if subsequent_latencies
        else None
    )

    mean_latency = sum(latencies) / len(latencies)

    latency_summary = {

        "first_measured_query_latency_ms": round(
            first_query_latency,
            2,
        ),

        "mean_subsequent_query_latency_ms": (
            round(mean_subsequent_latency, 2)
            if mean_subsequent_latency is not None
            else None
        ),

        "mean_latency_ms": round(
            mean_latency,
            2,
        ),

        "query_count": question_count,
    }

    # --------------------------------------------------
    # FINAL EVALUATION REPORT
    # --------------------------------------------------

    return {

        "dataset": selected_dataset.name,

        "evaluation_questions": question_count,

        "top_k": top_k,

        "rrf_k": rrf_k,

        "metrics": mean_metrics,

        "category_metrics": category_summary,

        "latency": latency_summary,

        "results": question_results,
    }


# --------------------------------------------------
# COMMAND-LINE EXECUTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Enterprise AI Knowledge Platform "
            "retrieval system."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["v1", "v2", "v3"],
        default="v1",
        help=(
            "Evaluation dataset: "
            "v1 (5 questions), "
            "v2 (50 questions), "
            "v3 (50 revised questions)."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to retrieve per question.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=(
            "Reciprocal Rank Fusion smoothing constant. "
            "Default: 40."
        ),
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    if args.rrf_k <= 0:
        parser.error("--rrf-k must be greater than zero.")

    # Dataset selection

    dataset_paths = {
        "v1": DEFAULT_DATASET_PATH,
        "v2": V2_DATASET_PATH,
        "v3": V3_DATASET_PATH,
    }

    dataset_path = dataset_paths[args.dataset]

    # Run evaluation

    report = evaluate_retrieval(
        top_k=args.top_k,
        dataset_path=dataset_path,
        rrf_k=args.rrf_k,
    )

    report["generated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    # Save report

    report_path = save_evaluation_report(report)

    # Display report

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print(
        f"\nEvaluation report saved to: {report_path}"
    )


if __name__ == "__main__":
    main()