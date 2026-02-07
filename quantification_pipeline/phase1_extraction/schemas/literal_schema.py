"""
Literal Track Schema - Pydantic models for structured output validation.

Based on phase1_literal_extraction_specialist.txt prompt requirements.
Enforces strict JSON schema compliance for LLM outputs.

Note: entities, actions, and relationships can be empty arrays.
object_id in relationships can be null for intransitive actions.
"""

from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class LiteralEntity(BaseModel):
    """Entity extracted from literal interpretation of idiom text."""
    
    id: str = Field(
        ...,
        pattern=r"^le_\d+$",
        description="Unique identifier in format 'le_X' where X is a number"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Noun from text or implied agent"
    )
    type: Literal["text_based", "placeholder", "null"] = Field(
        ...,
        description="text_based for nouns from text, placeholder for implied agents, null if empty"
    )


class LiteralAction(BaseModel):
    """Action (verb or preposition) from idiom text."""
    
    id: str = Field(
        ...,
        pattern=r"^la_\d+$",
        description="Unique identifier in format 'la_X' where X is a number"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Verb or preposition from text (e.g., 'Break', 'Under')"
    )


class LiteralRelationship(BaseModel):
    """Relationship linking subject, action, and object via IDs."""
    
    subject_id: str = Field(
        ...,
        pattern=r"^le_\d+$",
        description="Entity ID of the subject (who does it)"
    )
    action_id: str = Field(
        ...,
        pattern=r"^la_\d+$",
        description="Action ID reference"
    )
    object_id: Optional[str] = Field(
        None,
        pattern=r"^le_\d+$",
        description="Entity ID of the object (to whom/what). Can be null for intransitive actions."
    )


class LiteralTrack(BaseModel):
    """Complete literal scene graph for an idiom."""
    
    entities: List[LiteralEntity] = Field(
        default_factory=list,
        description="List of entities extracted from idiom. Can be empty if no entities found."
    )
    actions: List[LiteralAction] = Field(
        default_factory=list,
        description="List of actions/verbs/prepositions. Can be empty if no actions found."
    )
    relationships: List[LiteralRelationship] = Field(
        default_factory=list,
        description="Relationships linking entities via actions. Can be empty if no relationships found."
    )


class LiteralExtractionResult(BaseModel):
    """Root model for literal extraction output."""
    
    literal_track: LiteralTrack = Field(
        ...,
        description="The complete literal scene graph"
    )
