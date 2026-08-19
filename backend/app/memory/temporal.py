from datetime import datetime
from typing import Optional

from app.memory.graph import MemoryGraph
from app.models.association import Association


class TemporalMemoryEngine:

    def __init__(self, graph: MemoryGraph):
        self.graph = graph

    def supersede(
        self,
        old_memory_id: str,
        new_memory_id: str,
        strength: float = 1.0,
    ) -> Association:

        old_memory = self.graph.get_memory(
            old_memory_id
        )

        new_memory = self.graph.get_memory(
            new_memory_id
        )

        now = datetime.utcnow()

        # Old memory is no longer the current state.
        old_memory.valid_until = now
        old_memory.status = "superseded"

        # New memory becomes current.
        if new_memory.valid_from is None:
            new_memory.valid_from = now

        new_memory.status = "active"

        association = Association(
            id=(
                f"temporal_{old_memory_id}"
                f"_{new_memory_id}"
            ),
            source_id=old_memory_id,
            target_id=new_memory_id,
            relationship_type="supersedes",
            strength=strength,
        )

        self.graph.add_association(
            association
        )

        return association

    def contradict(
        self,
        memory_a_id: str,
        memory_b_id: str,
        strength: float = 1.0,
    ) -> Association:

        association = Association(
            id=(
                f"contradiction_"
                f"{memory_a_id}_"
                f"{memory_b_id}"
            ),
            source_id=memory_a_id,
            target_id=memory_b_id,
            relationship_type="contradicts",
            strength=strength,
        )

        self.graph.add_association(
            association
        )

        return association

    def confirm(
        self,
        older_memory_id: str,
        newer_memory_id: str,
        strength: float = 1.0,
    ) -> Association:

        association = Association(
            id=(
                f"confirmation_"
                f"{older_memory_id}_"
                f"{newer_memory_id}"
            ),
            source_id=older_memory_id,
            target_id=newer_memory_id,
            relationship_type="confirms",
            strength=strength,
        )

        self.graph.add_association(
            association
        )

        return association

    def expire(
        self,
        memory_id: str,
    ):

        memory = self.graph.get_memory(
            memory_id
        )

        memory.valid_until = datetime.utcnow()
        memory.status = "expired"

        return memory