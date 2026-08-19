from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class Entity(BaseModel):
    id: str
    name: str
    entity_type: str = "unknown"

    aliases: List[str] = Field(default_factory=list)

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )