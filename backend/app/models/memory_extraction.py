from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractedMemory(BaseModel):
    content: str
    memory_type: str

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    entities: List[str] = []
    topics: List[str] = []

    source_text: str

    reasoning: Optional[str] = None