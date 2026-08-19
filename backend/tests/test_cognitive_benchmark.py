from app.memory.activation import ActivationEngine
from app.memory.baseline import BaselineRetriever
from app.memory.retrieval import RetrievalEngine
from app.memory.graph import MemoryGraph
from app.models.association import Association
from app.models.memory_extraction import ExtractedMemory


def make_memory(
    content: str,
    entities: list[str],
    topics: list[str],
):
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

    # -----------------------------
    # Core memories
    # -----------------------------

    graph.add_memory(
        make_memory(
            "I require suppliers to deliver within 14 days.",
            ["User"],
            ["supplier requirements", "delivery"],
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Alpha usually delivers in 9 days.",
            ["Supplier Alpha"],
            ["suppliers", "delivery"],
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Beta usually takes 22 days.",
            ["Supplier Beta"],
            ["suppliers", "delivery"],
        )
    )

    graph.add_memory(
        make_memory(
            "I rejected Supplier Beta because they were too slow.",
            ["Supplier Beta"],
            ["suppliers", "delivery", "rejection"],
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Alpha increased their prices.",
            ["Supplier Alpha"],
            ["suppliers", "pricing"],
        )
    )

    # -----------------------------
    # Semantic relationships
    # -----------------------------

    graph.add_association(
        Association(
            id="a1",
            source_id="memory_1",
            target_id="memory_2",
            relationship_type="constrains",
            strength=0.95,
        )
    )

    graph.add_association(
        Association(
            id="a2",
            source_id="memory_1",
            target_id="memory_3",
            relationship_type="constrains",
            strength=0.95,
        )
    )

    graph.add_association(
        Association(
            id="a3",
            source_id="memory_3",
            target_id="memory_4",
            relationship_type="caused",
            strength=0.9,
        )
    )

    graph.add_association(
        Association(
            id="a4",
            source_id="memory_2",
            target_id="memory_5",
            relationship_type="related_to",
            strength=0.8,
        )
    )

    return graph


def get_rank(results, memory_id):

    for rank, (result_id, score) in enumerate(
        results,
        start=1,
    ):
        if result_id == memory_id:
            return rank

    return None


def test_direct_recall():

    graph = build_graph()

    activation = ActivationEngine(graph)
    cognitive = RetrievalEngine(graph, activation)

    results = cognitive.retrieve(
        "Supplier Alpha delivers in 9 days.",
        top_k=5,
    )

    assert get_rank(results, "memory_2") is not None


def test_associative_recall():

    graph = build_graph()

    activation = ActivationEngine(graph)
    cognitive = RetrievalEngine(graph, activation)

    results = cognitive.retrieve(
        "Which supplier should I consider?",
        top_k=5,
    )

    # The requirement should be discoverable through
    # the supplier concept and graph relationships.
    assert get_rank(results, "memory_1") is not None


def test_constraint_recall():

    graph = build_graph()

    activation = ActivationEngine(graph)
    cognitive = RetrievalEngine(graph, activation)

    results = cognitive.retrieve(
        "What delivery rules do I have?",
        top_k=5,
    )

    assert get_rank(results, "memory_1") is not None


def test_causal_recall():

    graph = build_graph()

    activation = ActivationEngine(graph)
    cognitive = RetrievalEngine(graph, activation)

    results = cognitive.retrieve(
        "Why did I reject Supplier Beta?",
        top_k=5,
    )

    assert get_rank(results, "memory_4") is not None


def test_distractor_resistance():

    graph = build_graph()

    activation = ActivationEngine(graph)
    cognitive = RetrievalEngine(graph, activation)

    results = cognitive.retrieve(
        "Why was Supplier Beta rejected?",
        top_k=1,
    )

    # The direct rejection memory should be the
    # strongest result, not an unrelated supplier memory.
    assert results[0][0] == "memory_4"