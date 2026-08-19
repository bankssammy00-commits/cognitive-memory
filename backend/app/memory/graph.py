from typing import Dict, List

from app.models.association import Association
from app.models.memory_extraction import ExtractedMemory


class MemoryGraph:

    def __init__(self):
        self.memories: Dict[str, ExtractedMemory] = {}
        self.associations: Dict[str, Association] = {}

    def add_memory(self, memory: ExtractedMemory) -> None:
        memory_id = f"memory_{len(self.memories) + 1}"

        self.memories[memory_id] = memory

    def add_association(
        self,
        association: Association,
    ) -> None:

        self.associations[association.id] = association

    def get_memory(
        self,
        memory_id: str,
    ) -> ExtractedMemory | None:

        return self.memories.get(memory_id)

    def get_associations(
        self,
        memory_id: str,
    ) -> List[Association]:

        return [
            association
            for association in self.associations.values()
            if (
                association.source_id == memory_id
                or association.target_id == memory_id
            )
        ]

    def memory_count(self) -> int:
        return len(self.memories)

    def association_count(self) -> int:
        return len(self.associations)