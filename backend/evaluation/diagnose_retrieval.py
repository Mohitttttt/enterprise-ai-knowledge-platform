import json
from pathlib import Path

from app.services.hybrid_search import HybridSearch
from app.services.vector_store import VectorStore


DATASET_PATH = (
    Path(__file__).parent
    / "dataset"
    / "retrieval_questions_v3.json"
)

TARGET_IDS = {"Q013", "Q040", "Q046", "Q048", "Q050"}

TOP_K = 5
RRF_K = 40


# --------------------------------------------------
# DISPLAY RETRIEVAL CANDIDATES
# --------------------------------------------------

def display_candidates(title, results, limit=5):

    print(f"\n{title}")

    for result in results[:limit]:

        metadata = result.get("metadata") or {}

        document = result.get("document", "")
        snippet = " ".join(document.split())

        print(f"\n  ID: {result['id']}")
        print(f"  Metadata: {metadata}")
        print(f"  Text: {snippet[:500]}")


# --------------------------------------------------
# MAIN DIAGNOSTIC RUNNER
# --------------------------------------------------

def main():

    # Load evaluation dataset
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = [
        item for item in dataset
        if item["id"] in TARGET_IDS
    ]

    if len(questions) != len(TARGET_IDS):
        raise RuntimeError(
            "Some target questions are missing."
        )

    # Connect to isolated evaluation collection
    store = VectorStore(
        collection_name="enterprise_evaluation"
    )

    if store.collection.count() == 0:
        raise RuntimeError(
            "Evaluation collection is empty."
        )

    # Initialize hybrid search
    search_service = HybridSearch(
        rrf_k=RRF_K,
        store=store,
    )

    print("=" * 90)
    print("RETRIEVAL DIAGNOSTICS")
    print("Collection: enterprise_evaluation")
    print(f"RRF K: {RRF_K}")
    print(f"Questions: {len(questions)}")
    print("=" * 90)

    # Process each target question
    for question in questions:

        question_id = question["id"]
        query = question["question"]
        relevant_ids = question["relevant_chunk_ids"]

        results = search_service.search_with_diagnostics(
            query=query,
            top_k=TOP_K,
        )

        semantic = results["semantic_results"]
        keyword = results["keyword_results"]
        fused = results["fused_results"]

        # Build rank lookup tables
        semantic_ranks = {
            item["id"]: item["semantic_rank"]
            for item in semantic
        }

        keyword_ranks = {
            item["id"]: item["keyword_rank"]
            for item in keyword
        }

        fused_ranks = {
            item["id"]: item["final_rank"]
            for item in fused
        }

        print("\n" + "=" * 90)
        print(f"{question_id}: {query}")
        print("=" * 90)

        # ------------------------------------------
        # RELEVANT CHUNK DIAGNOSTICS
        # ------------------------------------------

        print("\nRELEVANT CHUNK DIAGNOSTICS")

        for chunk_id in relevant_ids:

            semantic_match = next(
                (
                    item for item in semantic
                    if item["id"] == chunk_id
                ),
                None,
            )

            keyword_match = next(
                (
                    item for item in keyword
                    if item["id"] == chunk_id
                ),
                None,
            )

            fused_match = next(
                (
                    item for item in fused
                    if item["id"] == chunk_id
                ),
                None,
            )

            print(f"\nChunk: {chunk_id}")

            print(
                "  Semantic rank:",
                semantic_ranks.get(
                    chunk_id,
                    "NOT RETRIEVED",
                ),
            )

            if semantic_match:
                print(
                    "  Semantic distance:",
                    round(
                        semantic_match["distance"],
                        5,
                    ),
                )

            print(
                "  BM25 rank:",
                keyword_ranks.get(
                    chunk_id,
                    "NOT RETRIEVED",
                ),
            )

            if keyword_match:
                print(
                    "  BM25 score:",
                    round(
                        keyword_match["score"],
                        5,
                    ),
                )

            print(
                "  Fused rank:",
                fused_ranks.get(
                    chunk_id,
                    "NOT RETRIEVED",
                ),
            )

            if fused_match:

                print(
                    "  Semantic RRF contribution:",
                    round(
                        fused_match[
                            "semantic_rrf_contribution"
                        ],
                        6,
                    ),
                )

                print(
                    "  Keyword RRF contribution:",
                    round(
                        fused_match[
                            "keyword_rrf_contribution"
                        ],
                        6,
                    ),
                )

                print(
                    "  Total RRF score:",
                    round(
                        fused_match["rrf_score"],
                        6,
                    ),
                )

        # ------------------------------------------
        # DISPLAY RETRIEVAL RESULTS
        # ------------------------------------------

        display_candidates(
            "TOP SEMANTIC RESULTS",
            semantic,
        )

        display_candidates(
            "TOP BM25 RESULTS",
            keyword,
        )

        display_candidates(
            "TOP 10 FUSED RESULTS",
            fused,
            limit=10,
        )

    print("\n" + "=" * 90)
    print("DIAGNOSTICS COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()