import json
import math
from pathlib import Path
from statistics import mean

from app.services.keyword_search import KeywordSearch, tokenize
from app.services.vector_store import VectorStore


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    Path(__file__).parent
    / "dataset"
    / "retrieval_questions_v3.json"
)

COLLECTION_NAME = "enterprise_evaluation"

# None means evaluate all questions.
# Example: TARGET_IDS = {"Q013", "Q040"}
TARGET_IDS = None

TOP_K = 5


# ============================================================
# CONTROLLED SYNONYM EXPANSION DICTIONARY
# ============================================================

SYNONYMS = {
    "laptop": ["laptops", "device", "devices"],
    "monitor": ["monitors"],
    "stops": ["problem", "problems", "malfunction"],
    "breaks": [
        "problem",
        "problems",
        "malfunction",
        "repair",
        "replacement",
    ],
    "issue": ["problem", "problems"],
    "reporting": ["report", "submit", "submission"],
}


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query: str) -> list[str]:
    original_tokens = tokenize(query)

    # Preserve all original tokens, including repetitions.
    expanded_tokens = list(original_tokens)

    # Track existing tokens to prevent duplicate synonyms.
    existing_tokens = set(original_tokens)

    for token in original_tokens:
        for synonym in SYNONYMS.get(token, []):

            if synonym not in existing_tokens:
                expanded_tokens.append(synonym)
                existing_tokens.add(synonym)

    return expanded_tokens


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def retrieve(bm25, records, query_tokens):
    scores = bm25.get_scores(query_tokens)

    results = []

    for index, score in enumerate(scores):

        if score <= 0:
            continue

        document_id, document, metadata = records[index]

        results.append({
            "id": document_id,
            "score": float(score),
            "document": document,
            "metadata": metadata or {},
        })

    # Sort by descending BM25 score.
    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:TOP_K]


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(results, question):

    relevant_ids = set(question["relevant_chunk_ids"])
    grades = question["relevance_grades"]

    retrieved_ids = [
        result["id"]
        for result in results
    ]

    relevant_retrieved = [
        document_id
        for document_id in retrieved_ids
        if document_id in relevant_ids
    ]

    # --------------------------------------------------------
    # Precision@K
    # --------------------------------------------------------

    precision = len(relevant_retrieved) / TOP_K

    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    recall = (
        len(relevant_retrieved) / len(relevant_ids)
        if relevant_ids
        else 0
    )

    # --------------------------------------------------------
    # Mean Reciprocal Rank (MRR)
    # --------------------------------------------------------

    reciprocal_rank = 0

    for rank, document_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if document_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    # --------------------------------------------------------
    # Discounted Cumulative Gain (DCG)
    # --------------------------------------------------------

    dcg = sum(
        (
            2 ** grades.get(document_id, 0) - 1
        ) / math.log2(rank + 1)

        for rank, document_id in enumerate(
            retrieved_ids,
            start=1,
        )
    )

    # --------------------------------------------------------
    # Ideal DCG (IDCG)
    # --------------------------------------------------------

    ideal_grades = sorted(
        grades.values(),
        reverse=True,
    )[:TOP_K]

    idcg = sum(
        (2 ** grade - 1) / math.log2(rank + 1)

        for rank, grade in enumerate(
            ideal_grades,
            start=1,
        )
    )

    # --------------------------------------------------------
    # Normalized DCG (NDCG@K)
    # --------------------------------------------------------

    ndcg = (
        dcg / idcg
        if idcg
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
        "relevant_found": len(relevant_retrieved),
        "relevant_total": len(relevant_ids),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load evaluation dataset
    # --------------------------------------------------------

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(file)

    questions = [
        question
        for question in dataset
        if TARGET_IDS is None
        or question["id"] in TARGET_IDS
    ]

    if not questions:
        raise RuntimeError(
            "No evaluation questions were found."
        )

    # --------------------------------------------------------
    # 2. Connect to isolated evaluation collection
    # --------------------------------------------------------

    store = VectorStore(
        collection_name=COLLECTION_NAME
    )

    if store.collection.count() == 0:
        raise RuntimeError(
            "Evaluation collection is empty."
        )

    # --------------------------------------------------------
    # 3. Initialize BM25 search
    # --------------------------------------------------------

    keyword_service = KeywordSearch(
        store=store
    )

    bm25, records = keyword_service._get_index()

    if bm25 is None:
        raise RuntimeError(
            "BM25 index could not be created."
        )

    if not records:
        raise RuntimeError(
            "No records were available for BM25 retrieval."
        )

    # --------------------------------------------------------
    # 4. Initialize metric storage
    # --------------------------------------------------------

    baseline_metrics = []
    expanded_metrics = []

    # --------------------------------------------------------
    # 5. Print experiment configuration
    # --------------------------------------------------------

    print("=" * 90)
    print("BM25 QUERY EXPANSION EXPERIMENT")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Questions: {len(questions)}")
    print(f"Top K: {TOP_K}")
    print("=" * 90)

    # --------------------------------------------------------
    # 6. Evaluate every question
    # --------------------------------------------------------

    for question in questions:

        question_id = question["id"]
        query = question["question"]

        # Original query tokens.
        original_tokens = tokenize(query)

        # Expanded query tokens.
        expanded_tokens = expand_query(query)

        # ----------------------------------------------------
        # Baseline retrieval
        # ----------------------------------------------------

        baseline_results = retrieve(
            bm25,
            records,
            original_tokens,
        )

        # ----------------------------------------------------
        # Expanded retrieval
        # ----------------------------------------------------

        expanded_results = retrieve(
            bm25,
            records,
            expanded_tokens,
        )

        # ----------------------------------------------------
        # Calculate baseline metrics
        # ----------------------------------------------------

        baseline = calculate_metrics(
            baseline_results,
            question,
        )

        # ----------------------------------------------------
        # Calculate expanded metrics
        # ----------------------------------------------------

        expanded = calculate_metrics(
            expanded_results,
            question,
        )

        baseline_metrics.append(baseline)
        expanded_metrics.append(expanded)

        # ----------------------------------------------------
        # Display question details
        # ----------------------------------------------------

        print(f"\n{'-' * 90}")
        print(f"{question_id}: {query}")

        print("\nOriginal tokens:")
        print(original_tokens)

        print("\nExpanded tokens:")
        print(expanded_tokens)

        # ----------------------------------------------------
        # Display baseline results
        # ----------------------------------------------------

        print("\nBASELINE TOP RESULTS:")

        for rank, result in enumerate(
            baseline_results,
            start=1,
        ):

            print(
                f"{rank}. {result['id']} "
                f"| Score: {result['score']:.4f}"
            )

        # ----------------------------------------------------
        # Display expanded results
        # ----------------------------------------------------

        print("\nEXPANDED TOP RESULTS:")

        for rank, result in enumerate(
            expanded_results,
            start=1,
        ):

            print(
                f"{rank}. {result['id']} "
                f"| Score: {result['score']:.4f}"
            )

        # ----------------------------------------------------
        # Display metric comparison
        # ----------------------------------------------------

        print("\nMETRIC COMPARISON:")

        for metric in [
            "precision",
            "recall",
            "mrr",
            "ndcg",
        ]:

            print(
                f"{metric:12} "
                f"Baseline: {baseline[metric]:.4f} "
                f"| Expanded: {expanded[metric]:.4f}"
            )

        print(
            "Relevant chunks found: "
            f"{baseline['relevant_found']}/"
            f"{baseline['relevant_total']} "
            "-> "
            f"{expanded['relevant_found']}/"
            f"{expanded['relevant_total']}"
        )

    # ========================================================
    # 7. OVERALL RESULTS
    # ========================================================

    print("\n" + "=" * 90)
    print("OVERALL RESULTS")
    print("=" * 90)

    for metric in [
        "precision",
        "recall",
        "mrr",
        "ndcg",
    ]:

        baseline_average = mean(
            item[metric]
            for item in baseline_metrics
        )

        expanded_average = mean(
            item[metric]
            for item in expanded_metrics
        )

        difference = (
            expanded_average - baseline_average
        )

        print(
            f"{metric:12} "
            f"Baseline: {baseline_average:.4f} "
            f"| Expanded: {expanded_average:.4f} "
            f"| Difference: {difference:+.4f}"
        )

    # ========================================================
    # 8. SAFETY CONFIRMATION
    # ========================================================

    print("\nProduction code was not modified.")
    print("Production ChromaDB collection was not modified.")

    print("\nExperiment completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()