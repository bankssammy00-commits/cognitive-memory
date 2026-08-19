from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from app.memory.graph import MemoryGraph


# =========================================================
# Association Learning
# =========================================================

class AssociationLearner:
    """
    Adjusts association strength based on experience.

    Positive reward strengthens an association.
    Negative reward weakens an association.

    Strength is always constrained to [0.0, 1.0].
    """

    def __init__(
        self,
        graph: MemoryGraph,
        learning_rate: float = 0.10,
    ):
        self.graph = graph
        self.learning_rate = learning_rate

    # -----------------------------------------------------
    # Reinforcement
    # -----------------------------------------------------

    def reinforce(
        self,
        association_id: str,
        reward: float = 1.0,
    ) -> bool:

        association = self.graph.associations.get(
            association_id
        )

        if association is None:
            return False

        change = (
            self.learning_rate
            * max(0.0, reward)
        )

        association.strength = min(
            1.0,
            association.strength + change,
        )

        return True

    # -----------------------------------------------------
    # Weakening
    # -----------------------------------------------------

    def weaken(
        self,
        association_id: str,
        penalty: float = 1.0,
    ) -> bool:

        association = self.graph.associations.get(
            association_id
        )

        if association is None:
            return False

        change = (
            self.learning_rate
            * max(0.0, penalty)
        )

        association.strength = max(
            0.0,
            association.strength - change,
        )

        return True

    # -----------------------------------------------------
    # Unified learning
    # -----------------------------------------------------

    def learn(
        self,
        association_id: str,
        reward: float = 0.0,
    ) -> bool:

        if reward > 0:

            return self.reinforce(
                association_id,
                reward=reward,
            )

        if reward < 0:

            return self.weaken(
                association_id,
                penalty=abs(reward),
            )

        # Zero reward means no learning.
        return (
            association_id
            in self.graph.associations
        )


# =========================================================
# Retrieval Feedback
# =========================================================

@dataclass
class RetrievalFeedback:
    """
    Describes the outcome of a retrieval operation.
    """

    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    reward: float = 0.0


# =========================================================
# Memory Learning Statistics
# =========================================================

@dataclass
class MemoryLearningStats:

    successes: int = 0
    failures: int = 0
    reward: float = 0.0

    @property
    def total(self) -> int:
        return (
            self.successes
            + self.failures
        )

    @property
    def success_rate(self) -> float:

        if self.total == 0:
            return 0.0

        return (
            self.successes
            / self.total
        )


# =========================================================
# Relationship Learning Statistics
# =========================================================

@dataclass
class RelationshipLearningStats:

    successes: int = 0
    failures: int = 0
    reward: float = 0.0

    @property
    def total(self) -> int:
        return (
            self.successes
            + self.failures
        )

    @property
    def success_rate(self) -> float:

        if self.total == 0:
            return 0.0

        return (
            self.successes
            / self.total
        )


# =========================================================
# Learning Engine
# =========================================================

class LearningEngine:
    """
    Higher-level learning system.

    AssociationLearner modifies the graph directly.

    LearningEngine keeps experience statistics that can later
    influence retrieval and association formation.
    """

    def __init__(
        self,
        graph: Optional[MemoryGraph] = None,
        learning_rate: float = 0.10,
    ):
        self.graph = graph

        self.association_learner = (
            AssociationLearner(
                graph,
                learning_rate=learning_rate,
            )
            if graph is not None
            else None
        )

        self.memory_stats: Dict[
            str,
            MemoryLearningStats,
        ] = {}

        self.relationship_stats: Dict[
            str,
            RelationshipLearningStats,
        ] = {}

        self.query_stats: Dict[
            str,
            float,
        ] = {}

        self.total_feedback_events = 0

    # =====================================================
    # Memory statistics
    # =====================================================

    def _memory_stats(
        self,
        memory_id: str,
    ) -> MemoryLearningStats:

        if memory_id not in self.memory_stats:

            self.memory_stats[memory_id] = (
                MemoryLearningStats()
            )

        return self.memory_stats[
            memory_id
        ]

    def record_memory_outcome(
        self,
        memory_id: str,
        success: bool,
        reward: float = 1.0,
    ) -> None:

        stats = self._memory_stats(
            memory_id
        )

        if success:
            stats.successes += 1
        else:
            stats.failures += 1

        stats.reward += reward

    # =====================================================
    # Relationship statistics
    # =====================================================

    def _relationship_stats(
        self,
        relationship_id: str,
    ) -> RelationshipLearningStats:

        if (
            relationship_id
            not in self.relationship_stats
        ):

            self.relationship_stats[
                relationship_id
            ] = RelationshipLearningStats()

        return self.relationship_stats[
            relationship_id
        ]

    def record_relationship_outcome(
        self,
        relationship_id: str,
        success: bool,
        reward: float = 1.0,
    ) -> None:

        stats = self._relationship_stats(
            relationship_id
        )

        if success:
            stats.successes += 1
        else:
            stats.failures += 1

        stats.reward += reward

    # =====================================================
    # Query statistics
    # =====================================================

    @staticmethod
    def normalize_query(
        query: str,
    ) -> str:

        return " ".join(
            query.lower()
            .strip()
            .split()
        )

    def record_query_outcome(
        self,
        query: str,
        reward: float,
    ) -> None:

        normalized = (
            self.normalize_query(query)
        )

        self.query_stats[normalized] = (
            self.query_stats.get(
                normalized,
                0.0,
            )
            + reward
        )

    # =====================================================
    # Learn from retrieval
    # =====================================================

    def learn(
        self,
        feedback: RetrievalFeedback,
    ) -> None:

        self.total_feedback_events += 1

        relevant = set(
            feedback.relevant_ids
        )

        retrieved = set(
            feedback.retrieved_ids
        )

        # ---------------------------------------------
        # Query-level learning
        # ---------------------------------------------

        self.record_query_outcome(
            feedback.query,
            feedback.reward,
        )

        # ---------------------------------------------
        # Memory-level learning
        # ---------------------------------------------

        for memory_id in retrieved:

            is_relevant = (
                memory_id in relevant
            )

            if is_relevant:

                reward = abs(
                    feedback.reward
                )

            else:

                reward = -abs(
                    feedback.reward
                )

            self.record_memory_outcome(
                memory_id,
                success=is_relevant,
                reward=reward,
            )

    # =====================================================
    # Memory ranking adjustment
    # =====================================================

    def memory_boost(
        self,
        memory_id: str,
        max_boost: float = 0.25,
    ) -> float:

        stats = self.memory_stats.get(
            memory_id
        )

        if (
            stats is None
            or stats.total == 0
        ):
            return 0.0

        centered = (
            stats.success_rate * 2.0
            - 1.0
        )

        boost = (
            centered
            * max_boost
        )

        return max(
            -max_boost,
            min(
                max_boost,
                boost,
            ),
        )

    # =====================================================
    # Relationship ranking adjustment
    # =====================================================

    def relationship_boost(
        self,
        relationship_id: str,
        max_boost: float = 0.20,
    ) -> float:

        stats = self.relationship_stats.get(
            relationship_id
        )

        if (
            stats is None
            or stats.total == 0
        ):
            return 0.0

        centered = (
            stats.success_rate * 2.0
            - 1.0
        )

        boost = (
            centered
            * max_boost
        )

        return max(
            -max_boost,
            min(
                max_boost,
                boost,
            ),
        )

    # =====================================================
    # Query adjustment
    # =====================================================

    def query_boost(
        self,
        query: str,
        scale: float = 0.05,
    ) -> float:

        normalized = (
            self.normalize_query(query)
        )

        reward = self.query_stats.get(
            normalized,
            0.0,
        )

        return max(
            -scale,
            min(
                scale,
                reward * scale,
            ),
        )

    # =====================================================
    # Inspection
    # =====================================================

    def get_memory_stats(
        self,
        memory_id: str,
    ) -> Optional[MemoryLearningStats]:

        return self.memory_stats.get(
            memory_id
        )

    def get_relationship_stats(
        self,
        relationship_id: str,
    ) -> Optional[
        RelationshipLearningStats
    ]:

        return self.relationship_stats.get(
            relationship_id
        )

    # =====================================================
    # Reset
    # =====================================================

    def reset(self) -> None:

        self.memory_stats.clear()
        self.relationship_stats.clear()
        self.query_stats.clear()

        self.total_feedback_events = 0