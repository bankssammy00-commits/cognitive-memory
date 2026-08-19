from app.memory.activation import ActivationEngine
from app.memory.association_discovery import AssociationDiscovery
from app.memory.baseline import BaselineRetriever
from app.memory.graph import MemoryGraph
from app.memory.retrieval import RetrievalEngine
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

    # IMPORTANT:
    # No manual associations.

    discovery = AssociationDiscovery(
        graph,
        min_strength=0.20,
    )

    discovered = discovery.build_graph()

    print(
        f"Automatically discovered "
        f"{len(discovered)} associations."
    )

    for association in discovered:

        print(
            f"{association.source_id} -> "
            f"{association.target_id} | "
            f"{association.relationship_type} | "
            f"{association.strength:.3f}"
        )

    return graph


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


def evaluate(results, relevant):

    ranks = []

    for rank, (memory_id, score) in enumerate(
        results,
        start=1,
    ):
        if memory_id in relevant:
            ranks.append(rank)

    if not ranks:
        return {
            "r1": 0,
            "r3": 0,
            "r5": 0,
            "coverage": 0,
            "mrr": 0,
        }

    best = min(ranks)

    return {
        "r1": int(best <= 1),
        "r3": int(best <= 3),
        "r5": int(best <= 5),
        "coverage": len(ranks) / len(relevant),
        "mrr": 1 / best,
    }


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
    print("AUTOMATIC COGNITIVE MEMORY BENCHMARK")
    print("====================================")
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

        b = evaluate(
            baseline_results,
            relevant,
        )

        c = evaluate(
            cognitive_results,
            relevant,
        )

        baseline_scores.append(b)
        cognitive_scores.append(c)

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
            f"Baseline: "
            f"{[x[0] for x in baseline_results]}"
        )

        print(
            f"Cognitive: "
            f"{[x[0] for x in cognitive_results]}"
        )

        print(
            f"Baseline coverage: "
            f"{b['coverage']:.2f}"
        )

        print(
            f"Cognitive coverage: "
            f"{c['coverage']:.2f}"
        )

        print()

    def avg(scores, key):
        return sum(
            x[key]
            for x in scores
        ) / len(scores)

    print()
    print("FINAL RESULTS")
    print("=============")
    print()

    for key, label in [
        ("r1", "Recall@1"),
        ("r3", "Recall@3"),
        ("r5", "Recall@5"),
        ("coverage", "Coverage"),
        ("mrr", "MRR"),
    ]:

        b = avg(
            baseline_scores,
            key,
        )

        c = avg(
            cognitive_scores,
            key,
        )

        print(label)

        print(
            f"  Baseline:  {b:.3f}"
        )

        print(
            f"  Cognitive: {c:.3f}"
        )

        print(
            f"  Difference: {c - b:+.3f}"
        )

        print()


if __name__ == "__main__":
    run()