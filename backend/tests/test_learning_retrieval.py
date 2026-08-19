from app.memory.graph import MemoryGraph
from app.models.memory import Memory
from app.memory.activation import ActivationEngine
from app.memory.retrieval import RetrievalEngine
from app.memory.learning import (
    LearningEngine,
    RetrievalFeedback,
)


def make_memory(
    memory_id: str,
    content: str,
):
    return Memory(
        id=memory_id,
        content=content,
    )


def test_learning_tracks_retrieval_feedback():

    graph = MemoryGraph()

    graph.add_memory(
        make_memory(
            "memory_1",
            "Supplier Alpha delivers in 9 days.",
        )
    )

    graph.add_memory(
        make_memory(
            "memory_2",
            "Supplier Beta delivers in 22 days.",
        )
    )

    learning = LearningEngine(
        graph
    )

    feedback = RetrievalFeedback(
        query="Which supplier delivers quickly?",
        retrieved_ids=[
            "memory_1",
            "memory_2",
        ],
        relevant_ids=[
            "memory_1",
        ],
        reward=1.0,
    )

    learning.learn(feedback)

    alpha = learning.get_memory_stats(
        "memory_1"
    )

    beta = learning.get_memory_stats(
        "memory_2"
    )

    assert alpha is not None
    assert beta is not None

    assert alpha.successes == 1
    assert beta.failures == 1

    assert (
        learning.memory_boost("memory_1")
        > learning.memory_boost("memory_2")
    )


def test_query_learning_is_recorded():

    learning = LearningEngine()

    feedback = RetrievalFeedback(
        query="Which supplier delivers quickly?",
        retrieved_ids=[],
        relevant_ids=[],
        reward=1.0,
    )

    learning.learn(feedback)

    assert (
        learning.query_boost(
            "Which supplier delivers quickly?"
        )
        > 0
    )


def test_unseen_memory_has_no_learning_bias():

    learning = LearningEngine()

    assert (
        learning.memory_boost(
            "unknown_memory"
        )
        == 0.0
    )