from app.memory.consolidation import MemoryConsolidator
from app.memory.graph import MemoryGraph
from app.models.memory import Memory


def build_graph():
    graph = MemoryGraph()

    graph.add_memory(
        Memory(
            id="memory_1",
            content="Supplier Alpha delivers in 7 days.",
        )
    )

    graph.add_memory(
        Memory(
            id="memory_2",
            content="Supplier Beta delivers in 30 days.",
        )
    )

    return graph


def test_retrieval_is_recorded():
    graph = build_graph()

    consolidator = MemoryConsolidator(graph)

    assert consolidator.record_retrieval(
        "memory_1"
    )

    stats = consolidator.stats["memory_1"]

    assert stats.retrieval_count == 1
    assert stats.successful_retrievals == 1


def test_failed_retrieval_is_recorded():
    graph = build_graph()

    consolidator = MemoryConsolidator(graph)

    assert consolidator.record_retrieval(
        "memory_1",
        successful=False,
    )

    stats = consolidator.stats["memory_1"]

    assert stats.retrieval_count == 1
    assert stats.failed_retrievals == 1


def test_reinforcement_is_recorded():
    graph = build_graph()

    consolidator = MemoryConsolidator(graph)

    assert consolidator.record_reinforcement(
        "memory_1"
    )

    assert (
        consolidator.stats["memory_1"]
        .reinforcement_count
        == 1
    )


def test_reinforced_memory_becomes_more_important():
    graph = build_graph()

    consolidator = MemoryConsolidator(graph)

    before = consolidator.calculate_importance(
        "memory_1"
    )

    consolidator.record_retrieval(
        "memory_1"
    )

    consolidator.record_reinforcement(
        "memory_1"
    )

    after = consolidator.calculate_importance(
        "memory_1"
    )

    assert after > before


def test_importance_is_bounded():
    graph = build_graph()

    consolidator = MemoryConsolidator(graph)

    for _ in range(100):
        consolidator.record_retrieval(
            "memory_1"
        )
        consolidator.record_reinforcement(
            "memory_1"
        )

    importance = consolidator.calculate_importance(
        "memory_1"
    )

    assert 0.0 <= importance <= 1.0


def test_decay_never_goes_negative():
    graph = build_graph()

    consolidator = MemoryConsolidator(
        graph,
        decay_rate=1.0,
    )

    consolidator.importance["memory_1"] = 0.1

    result = consolidator.decay(
        "memory_1"
    )

    assert result >= 0.0


def test_reinforcement_protects_against_decay():
    graph = build_graph()

    consolidator = MemoryConsolidator(
        graph,
        decay_rate=0.5,
    )

    consolidator.importance["memory_1"] = 0.8

    normal = consolidator.decay(
        "memory_1"
    )

    consolidator.importance["memory_1"] = 0.8

    for _ in range(10):
        consolidator.record_reinforcement(
            "memory_1"
        )

    protected = consolidator.decay(
        "memory_1"
    )

    assert protected > normal


def test_missing_memory_is_safe():
    graph = build_graph()

    consolidator = MemoryConsolidator(graph)

    assert not consolidator.record_retrieval(
        "missing"
    )

    assert not consolidator.record_reinforcement(
        "missing"
    )

    assert (
        consolidator.calculate_importance("missing")
        == 0.0
    )


def test_low_importance_memory_can_be_archived():
    graph = build_graph()

    consolidator = MemoryConsolidator(
        graph,
        archive_threshold=0.15,
    )

    consolidator.importance["memory_2"] = 0.05

    assert consolidator.should_archive(
        "memory_2"
    )


def test_consolidation_returns_importance():
    graph = build_graph()

    consolidator = MemoryConsolidator(graph)

    score = consolidator.consolidate(
        "memory_1"
    )

    assert 0.0 <= score <= 1.0