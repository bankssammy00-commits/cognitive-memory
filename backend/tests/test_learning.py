from app.memory.graph import MemoryGraph
from app.models.association import Association
from app.memory.learning import AssociationLearner


def build_graph():
    graph = MemoryGraph()

    graph.add_association(
        Association(
            id="a1",
            source_id="memory_1",
            target_id="memory_2",
            relationship_type="related_to",
            strength=0.5,
        )
    )

    graph.add_association(
        Association(
            id="a2",
            source_id="memory_2",
            target_id="memory_3",
            relationship_type="causal",
            strength=0.5,
        )
    )

    return graph


def test_reinforce_strengthens_association():
    graph = build_graph()

    learner = AssociationLearner(
        graph,
        learning_rate=0.10,
    )

    learner.reinforce(
        "a1",
        reward=1.0,
    )

    association = graph.associations["a1"]

    assert association.strength > 0.5
    assert association.strength <= 1.0


def test_weaken_reduces_association():
    graph = build_graph()

    learner = AssociationLearner(
        graph,
        learning_rate=0.10,
    )

    learner.weaken(
        "a1",
        penalty=1.0,
    )

    association = graph.associations["a1"]

    assert association.strength < 0.5
    assert association.strength >= 0.0


def test_strength_never_exceeds_one():
    graph = build_graph()

    graph.associations["a1"].strength = 0.99

    learner = AssociationLearner(
        graph,
        learning_rate=1.0,
    )

    learner.reinforce(
        "a1",
        reward=1.0,
    )

    assert graph.associations["a1"].strength <= 1.0


def test_strength_never_goes_below_zero():
    graph = build_graph()

    graph.associations["a1"].strength = 0.01

    learner = AssociationLearner(
        graph,
        learning_rate=1.0,
    )

    learner.weaken(
        "a1",
        penalty=1.0,
    )

    assert graph.associations["a1"].strength >= 0.0


def test_learning_does_not_modify_unrelated_association():
    graph = build_graph()

    learner = AssociationLearner(
        graph,
        learning_rate=0.10,
    )

    original_strength = graph.associations["a2"].strength

    learner.reinforce(
        "a1",
        reward=1.0,
    )

    assert (
        graph.associations["a2"].strength
        == original_strength
    )


def test_unified_learning_positive():
    graph = build_graph()

    learner = AssociationLearner(
        graph,
        learning_rate=0.10,
    )

    original = graph.associations["a1"].strength

    assert learner.learn(
        "a1",
        reward=0.5,
    )

    assert (
        graph.associations["a1"].strength
        > original
    )


def test_unified_learning_negative():
    graph = build_graph()

    learner = AssociationLearner(
        graph,
        learning_rate=0.10,
    )

    original = graph.associations["a1"].strength

    assert learner.learn(
        "a1",
        reward=-0.5,
    )

    assert (
        graph.associations["a1"].strength
        < original
    )


def test_zero_reward_does_not_change_strength():
    graph = build_graph()

    learner = AssociationLearner(
        graph,
        learning_rate=0.10,
    )

    original = graph.associations["a1"].strength

    assert learner.learn(
        "a1",
        reward=0.0,
    )

    assert (
        graph.associations["a1"].strength
        == original
    )


def test_missing_association_is_safe():
    graph = build_graph()

    learner = AssociationLearner(graph)

    assert learner.reinforce("does_not_exist") is False
    assert learner.weaken("does_not_exist") is False