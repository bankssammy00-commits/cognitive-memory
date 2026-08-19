from datetime import datetime
from pydantic import BaseModel, Field


class Association(BaseModel):
    id: str

    source_id: str
    target_id: str

    relationship_type: str

    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    activation_count: int = 0