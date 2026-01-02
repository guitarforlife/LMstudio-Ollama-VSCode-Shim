"""Shared Pydantic model helpers."""

from pydantic import BaseModel, ConfigDict


class OllamaBaseModel(BaseModel):
    """Base model with permissive extra fields."""

    model_config = ConfigDict(extra="allow")
