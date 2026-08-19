from app.memory.activation import ActivationEngine
from app.memory.baseline import BaselineRetriever
from app.memory.retrieval import RetrievalEngine
from app.memory.graph import MemoryGraph
from app.models.association import Association
from app.models.memory_extraction import ExtractedMemory


def make_memory(content, entities, topics):
    return ExtractedMemory(
        content=content,
        memory_type="event",
        confidence=0.95,
        importance=0.8,
        entities=entities,
        topics=topics,
        source_text=content,
    )


def build_graph():

    graph = MemoryGraph()

    memories = [
        (
            "I require suppliers to deliver within 14 days.",
            ["User"],
            ["supplier", "delivery", "requirement"],
        ),
        (
            "Supplier Alpha usually delivers in 9 days.",
            ["Supplier Alpha"],
            ["supplier", "delivery"],
        ),
        (
            "Supplier Beta usually takes 22 days.",
            ["Supplier Beta"],
            ["supplier", "delivery"],
        ),
        (
            "I rejected Supplier Beta because they were too slow.",
            ["Supplier Beta"],
            ["supplier", "delivery", "rejection"],
        ),
        (
            "Supplier Alpha increased their prices.",
            ["Supplier Alpha"],
            ["supplier", "pricing"],
        ),
        (
            "I prefer reliable suppliers even when they cost slightly more.",
            ["User"],
            ["supplier", "preference", "pricing", "reliability"],
        ),
        (
            "Supplier Gamma delivers in 11 days.",
            ["Supplier Gamma"],
            ["supplier", "delivery"],
        ),
        (
            "Supplier Gamma has inconsistent quality.",
            ["Supplier Gamma"],
            ["supplier", "quality", "reliability"],
        ),
    ]

    for content, entities, topics in memories:
        graph.add_memory(
            make_memory(
                content,
                entities,
                topics,
            )
        )

    associations = [

        # Requirement -> supplier performance
        ("a1", "memory_1", "memory_2", "constrains", 0.95),
        ("a2", "memory_1", "memory_3", "constrains", 0.95),
        ("a3", "memory_1", "memory_7", "constrains", 0.95),

        # Beta performance -> rejection
        ("a4", "memory_3", "memory_4", "caused", 0.95),

        # Alpha -> pricing
        ("a5", "memory_2", "memory_5", "related_to", 0.80),

        # User preference -> reliability
        ("a6", "memory_6", "memory_8", "related_to", 0.85),

        # Gamma performance -> quality
        ("a7", "memory_7", "memory_8", "related_to", 0.80),

        # General preference -> Alpha
        ("a8", "memory_6", "memory_2", "supports", 0.70),
    ]

    for (
        association_id,
        source,
        target,
        relationship,
        strength,
    ) in associations:

        graph.add_association(
            Association(
                id=association_id,
                source_id=source,
                target_id=target,
                relationship_type=relationship,
                strength=strength,
            )
        )

    return graph


# ============================================================
# BENCHMARK CASES
# ============================================================

BENCHMARKS = [

    {
        "name": "direct_recall",
        "query": "How long does Supplier Alpha take?",
        "relevant": ["memory_2"],
    },

    {
        "name": "constraint_reasoning",
        "query": "Which supplier meets my 14 day delivery requirement?",
        "relevant": ["memory_1", "memory_2"],
    },

    {
        "name": "rejection_reasoning",
        "query": "Why was Supplier Beta rejected?",
        "relevant": ["memory_3", "memory_4"],
    },

    {
        "name": "multi_hop",
        "query": "Which supplier is likely a good choice based on my delivery requirement?",
        "relevant": ["memory_1", "memory_2"],
    },

    {
        "name": "preference_reasoning",
        "query": "What kind of supplier do I prefer?",
        "relevant": ["memory_6"],
    },

    {
        "name": "distractor_resistance",
        "query": "Which supplier is fast but has a reliability concern?",
        "relevant": ["memory_7", "memory_8"],
    },

    {
        "name": "pricing_context",
        "query": "What happened with Alpha after I learned they were fast?",
        "relevant": ["memory_2", "memory_5"],
    },
]


def rank_of(results, relevant_ids):

    ranks = []

    for rank, (memory_id, score) in enumerate(
        results,
        start=1,
    ):

        if memory_id in relevant_ids:
            ranks.append(rank)

    return ranks


def evaluate(results, relevant_ids):

    ranks = rank_of(
        results,
        relevant_ids,
    )

    if not ranks:
        return {
            "recall_1": False,
            "recall_3": False,
            "recall_5": False,
            "coverage": 0.0,
            "mrr": 0.0,
        }

    best_rank = min(ranks)

    retrieved_relevant = len(ranks)

    coverage = (
        retrieved_relevant
        / len(relevant_ids)
    )

    return {
        "recall_1": best_rank <= 1,
        "recall_3": best_rank <= 3,
        "recall_5": best_rank <= 5,
        "coverage": coverage,
        "mrr": 1.0 / best_rank,
    }


def average(scores, key):

    return sum(
        score[key]
        for score in scores
    ) / len(scores)


def run():

    graph = build_graph()

    activation = ActivationEngine(
        graph,
        decay=0.7,
        threshold=0.05,
    )

    cognitive = RetrievalEngine(
        graph,
        activation,
    )

    baseline = BaselineRetriever(
        graph
    )

    baseline_scores = []
    cognitive_scores = []

    print()
    print("COGNITIVE MEMORY BENCHMARK v2")
    print("==============================")
    print()

    for benchmark in BENCHMARKS:

        query = benchmark["query"]
        relevant = benchmark["relevant"]

        baseline_results = baseline.retrieve(
            query,
            top_k=8,
        )

        cognitive_results = cognitive.retrieve(
            query,
            top_k=8,
        )

        baseline_eval = evaluate(
            baseline_results,
            relevant,
        )

        cognitive_eval = evaluate(
            cognitive_results,
            relevant,
        )

        baseline_scores.append(
            baseline_eval
        )

        cognitive_scores.append(
            cognitive_eval
        )

        print(
            benchmark["name"]
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Expected: {relevant}"
        )

        print(
            f"Baseline: {rank_of(baseline_results, relevant)}"
        )

        print(
            f"Cognitive: {rank_of(cognitive_results, relevant)}"
        )

        print(
            f"Baseline coverage: "
            f"{baseline_eval['coverage']:.2f}"
        )

        print(
            f"Cognitive coverage: "
            f"{cognitive_eval['coverage']:.2f}"
        )

        print()

    print()
    print("FINAL RESULTS")
    print("=============")
    print()

    metrics = [
        ("recall_1", "Recall@1"),
        ("recall_3", "Recall@3"),
        ("recall_5", "Recall@5"),
        ("coverage", "Relevant-memory coverage"),
        ("mrr", "MRR"),
    ]

    for key, label in metrics:

        baseline_value = average(
            baseline_scores,
            key,
        )

        cognitive_value = average(
            cognitive_scores,
            key,
        )

        improvement = (
            cognitive_value
            - baseline_value
        )

        print(
            f"{label}:"
        )

        print(
            f"  Baseline:  {baseline_value:.3f}"
        )

        print(
            f"  Cognitive: {cognitive_value:.3f}"
        )

        print(
            f"  Difference: {improvement:+.3f}"
        )

        print()


if __name__ == "__main__":
    run()