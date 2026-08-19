from app.models.memory_extraction import ExtractedMemory


class MemoryValidator:

    MIN_CONFIDENCE = 0.70
    MIN_IMPORTANCE = 0.30

    def validate(self, memory: ExtractedMemory) -> bool:
        if memory.confidence < self.MIN_CONFIDENCE:
            return False

        if memory.importance < self.MIN_IMPORTANCE:
            return False

        if not memory.content.strip():
            return False

        if not memory.source_text.strip():
            return False

        return True