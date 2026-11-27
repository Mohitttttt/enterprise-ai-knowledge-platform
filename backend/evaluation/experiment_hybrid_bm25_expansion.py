import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.hybrid_search import HybridSearch
from app.services.keyword_search import KeywordSearch
from app.services.vector_store import VectorStore

from evaluation.evaluator import (
    evaluate_retrieval,
    save_evaluation_report,
)

from evaluation.experiment_bm25_expansion import (
    SYNONYMS,
    expand_query,
)


# ============================================================
# CONFIGURATION
# ============================================================

V4_DATASET_PATH = Path(__file__).parent / "dataset" / "retrieval_questions_v4.json"

COLLECTION_NAME = "enterprise_evaluation"
TOP_K = 5

REPORTS_DIR = Path(__file__).parent / "reports"


# ============================================================
# EXPERIMENTAL KEYWORD SEARCH
# ============================================================

class ExpandedKeywordSearch(KeywordSearch):
    """
    Experimental BM25 search with controlled query expansion.

    Reuses the existing KeywordSearch index and result format.
    Only the query tokens are expanded.
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not query or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        tokenized_query = expand_query(query)

        if not tokenized_query:
            return []

        bm25, valid_records = self._get_index()

        if bm25 is None or not valid_records:
            return []

        scores = bm25.get_scores(tokenized_query)

        ranked_results = []

        for index, score in enumerate(scores):

            if score <= 0:
                continue

            document_id, document, metadata = (
                valid_records[index]
            )

            ranked_results.append({
                "id": document_id,
                "document": document,
                "metadata": metadata or {},
                "score": float(score),
            })

        ranked_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked_results[:top_k]


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    name: str,
    store: VectorStore,
    rrf_k: int,
    use_expansion: bool,
) -> dict[str, Any]:

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {name}")
    print("=" * 65)

    if use_expansion:
        keyword_service = ExpandedKeywordSearch(store=store)
    else:
        keyword_service = KeywordSearch(store=store)

    search_service = HybridSearch(
        rrf_k=rrf_k,
        store=store,
        keyword_service=keyword_service,
    )

    report = evaluate_retrieval(
        search_service=search_service,
        top_k=TOP_K,
        dataset_path=V4_DATASET_PATH,
        rrf_k=rrf_k,
    )

    report["experiment_name"] = name
    report["collection"] = COLLECTION_NAME
    report["keyword_query_expansion"] = use_expansion

    if use_expansion:
        report["synonym_dictionary"] = SYNONYMS

    report["generated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    report_path = save_evaluation_report(report)

    print(f"\nReport saved: {report_path}")

    print("\nAggregate metrics:")

    for metric, value in report.get(
        "metrics", {}
    ).items():
        print(f"  {metric}: {value}")

    print("\nLatency:")

    for metric, value in report.get(
        "latency", {}
    ).items():
        print(f"  {metric}: {value}")

    return report


# ============================================================
# COMPARISON
# ============================================================

def build_comparison(
    baseline: dict[str, Any],
    expanded: dict[str, Any],
    k60: dict[str, Any],
) -> dict[str, Any]:

    configurations = {
        "baseline_rrf_40": baseline,
        "expanded_bm25_rrf_40": expanded,
        "baseline_rrf_60": k60,
    }

    comparison = {
        "experiment": "Hybrid BM25 Query Expansion",
        "collection": COLLECTION_NAME,
        "dataset": V4_DATASET_PATH.name,
        "top_k": TOP_K,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "configurations": {},
    }

    baseline_metrics = baseline.get(
        "metrics", {}
    )

    for name, report in configurations.items():

        metrics = report.get(
            "metrics", {}
        )

        deltas = {}

        if name != "baseline_rrf_40":

            for metric, value in metrics.items():

                baseline_value = baseline_metrics.get(metric)

                if (
                    isinstance(value, (int, float))
                    and isinstance(baseline_value, (int, float))
                ):
                    deltas[metric] = round(
                        value - baseline_value,
                        6,
                    )

        comparison["configurations"][name] = {
            "metrics": metrics,
            "latency": report.get("latency", {}),
            "delta_vs_baseline_rrf_40": deltas,
            "experiment_name": report.get("experiment_name"),
        }

    return comparison


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 65)
    print("HYBRID BM25 QUERY EXPANSION EXPERIMENT")
    print("=" * 65)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Dataset: {V4_DATASET_PATH}")
    print(f"Top K: {TOP_K}")

    store = VectorStore(
        collection_name=COLLECTION_NAME
    )

    collection_count = store.collection.count()

    if collection_count == 0:
        raise RuntimeError(
            "Evaluation collection is empty. "
            "Experiment cancelled."
        )

    print(f"Evaluation records: {collection_count}")

    baseline = run_experiment(
        name="Baseline Hybrid RRF K=40",
        store=store,
        rrf_k=40,
        use_expansion=False,
    )

    expanded = run_experiment(
        name="Expanded BM25 Hybrid RRF K=40",
        store=store,
        rrf_k=40,
        use_expansion=True,
    )

    k60 = run_experiment(
        name="Baseline Hybrid RRF K=60",
        store=store,
        rrf_k=60,
        use_expansion=False,
    )

    comparison = build_comparison(
        baseline,
        expanded,
        k60,
    )

    comparison_path = save_evaluation_report(
        comparison
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED")
    print("=" * 65)

    print(f"Comparison report: {comparison_path}")

    print("\nMetric differences versus baseline K=40:")

    for name, configuration in comparison[
        "configurations"
    ].items():

        print(f"\n{name}")

        for metric, delta in configuration[
            "delta_vs_baseline_rrf_40"
        ].items():
            print(f"  {metric}: {delta:+.6f}")


if __name__ == "__main__":
    main()


