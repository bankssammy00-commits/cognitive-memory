from app.memory.graph import MemoryGraph
from app.memory.temporal import TemporalMemoryEngine
from app.models.memory import Memory


def make_memory(
    memory_id: str,
    content: str,
):

    return Memory(
        id=memory_id,
        content=content,
        memory_type="preference",
        confidence=0.95,
        importance=0.8,
    )


def test_supersession():

    graph = MemoryGraph()

    old_memory = make_memory(
        "memory_old",
        "I prefer Supplier Alpha.",
    )

    new_memory = make_memory(
        "memory_new",
        "I stopped using Supplier Alpha.",
    )

    graph.memories[
        old_memory.id
    ] = old_memory

    graph.memories[
        new_memory.id
    ] = new_memory

    temporal = TemporalMemoryEngine(
        graph
    )

    association = temporal.supersede(
        "memory_old",
        "memory_new",
    )

    assert (
        association.relationship_type
        == "supersedes"
    )

    assert (
        old_memory.status
        == "superseded"
    )

    assert (
        old_memory.valid_until
        is not None
    )

    assert (
        new_memory.status
        == "active"
    )


def test_contradiction():

    graph = MemoryGraph()

    memory_a = make_memory(
        "memory_a",
        "Supplier Alpha delivers within 10 days.",
    )

    memory_b = make_memory(
        "memory_b",
        "Supplier Alpha now delivers in 20 days.",
    )

    graph.memories[
        memory_a.id
    ] = memory_a

    graph.memories[
        memory_b.id
    ] = memory_b

    temporal = TemporalMemoryEngine(
        graph
    )

    association = temporal.contradict(
        "memory_a",
        "memory_b",
    )

    assert (
        association.relationship_type
        == "contradicts"
    )

    assert (
        association.strength
        == 1.0
    )