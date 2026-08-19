from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.memory.graph import MemoryGraph


@dataclass
class MemoryStats:
    """
    Runtime statistics used to determine whether a memory
    should remain highly active, decay, or be archived.
    """

    retrieval_count: int = 0
    successful_retrievals: int = 0
    failed_retrievals: int = 0
    reinforcement_count: int = 0
    last_accessed: Optional[datetime] = None


class MemoryConsolidator:
    """
    Controls long-term memory consolidation.

    This first version is intentionally non-destructive:
    it calculates importance and decay pressure but does not
    delete memories.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        decay_rate: float = 0.01,
        archive_threshold: float = 0.15,
    ):
        self.graph = graph

        self.decay_rate = max(
            0.0,
            min(1.0, decay_rate),
        )

        self.archive_threshold = max(
            0.0,
            min(1.0, archive_threshold),
        )

        self.stats: dict[str, MemoryStats] = {}

        self.importance: dict[str, float] = {}

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def _get_stats(
        self,
        memory_id: str,
    ) -> MemoryStats:

        if memory_id not in self.stats:
            self.stats[memory_id] = MemoryStats()

        return self.stats[memory_id]

    # ---------------------------------------------------------
    # Retrieval tracking
    # ---------------------------------------------------------

    def record_retrieval(
        self,
        memory_id: str,
        successful: bool = True,
    ) -> bool:

        if self.graph.get_memory(memory_id) is None:
            return False

        stats = self._get_stats(memory_id)

        stats.retrieval_count += 1
        stats.last_accessed = datetime.utcnow()

        if successful:
            stats.successful_retrievals += 1
        else:
            stats.failed_retrievals += 1

        return True

    # ---------------------------------------------------------
    # Reinforcement tracking
    # ---------------------------------------------------------

    def record_reinforcement(
        self,
        memory_id: str,
    ) -> bool:

        if self.graph.get_memory(memory_id) is None:
            return False

        stats = self._get_stats(memory_id)

        stats.reinforcement_count += 1

        return True

    # ---------------------------------------------------------
    # Importance
    # ---------------------------------------------------------

    def calculate_importance(
        self,
        memory_id: str,
    ) -> float:

        if self.graph.get_memory(memory_id) is None:
            return 0.0

        stats = self._get_stats(memory_id)

        retrieval_score = min(
            stats.retrieval_count / 10.0,
            1.0,
        )

        success_score = 0.0

        if stats.retrieval_count > 0:
            success_score = (
                stats.successful_retrievals
                / stats.retrieval_count
            )

        reinforcement_score = min(
            stats.reinforcement_count / 5.0,
            1.0,
        )

        importance = (
            retrieval_score * 0.35
            + success_score * 0.35
            + reinforcement_score * 0.30
        )

        importance = max(
            0.0,
            min(1.0, importance),
        )

        self.importance[memory_id] = importance

        return importance

    # ---------------------------------------------------------
    # Decay
    # ---------------------------------------------------------

    def decay(
        self,
        memory_id: str,
    ) -> float:

        current = self.importance.get(
            memory_id,
            self.calculate_importance(memory_id),
        )

        stats = self._get_stats(memory_id)

        # Frequently reinforced memories resist decay.
        protection = min(
            stats.reinforcement_count / 10.0,
            1.0,
        )

        effective_decay = (
            self.decay_rate
            * (1.0 - protection)
        )

        new_importance = max(
            0.0,
            current - effective_decay,
        )

        self.importance[memory_id] = new_importance

        return new_importance

    # ---------------------------------------------------------
    # Consolidation
    # ---------------------------------------------------------

    def consolidate(
        self,
        memory_id: str,
    ) -> float:

        importance = self.calculate_importance(
            memory_id
        )

        # Important memories are reinforced rather than
        # immediately decayed.
        if importance >= 0.5:
            return importance

        return self.decay(memory_id)

    # ---------------------------------------------------------
    # Archive decision
    # ---------------------------------------------------------

    def should_archive(
        self,
        memory_id: str,
    ) -> bool:

        if self.graph.get_memory(memory_id) is None:
            return False

        importance = self.importance.get(
            memory_id,
            self.calculate_importance(memory_id),
        )

        return importance <= self.archive_threshold