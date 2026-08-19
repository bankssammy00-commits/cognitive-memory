from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Memory(BaseModel):
    id: str
    content: str

    memory_type: str = "episodic"

    created_at: datetime = Field(default_factory=datetime.utcnow)

    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    source_id: Optional[str] = None

    access_count: int = 0
    last_accessed: Optional[datetime] = None

    status: str = "active"