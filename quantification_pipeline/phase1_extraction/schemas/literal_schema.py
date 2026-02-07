"""
Literal Track Schema - Pydantic models for structured output validation.

Based on phase1_literal_extraction_specialist.txt prompt requirements.
Enforces strict JSON schema compliance for LLM outputs.

Note: 
- entities, actions must have at least 1 item (use null placeholder if truly empty)
- relationships can be empty if no relationships exist
- object_id in relationships can be null for intransitive actions
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
        description="Noun from text, implied agent, or 'null' if truly nothing found"
    )
    type: Literal["text_based", "placeholder", "null"] = Field(
        ...,
        description="text_based for nouns from text, placeholder for implied agents, null if nothing identifiable"
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Short rationale explaining why this entity is included"
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
        description="Verb or preposition from text, or 'null' if truly nothing found"
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Short rationale explaining why this action is included"
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
    
    thought_process: str = Field(
        ...,
        min_length=10,
        description="CoT-style summary: Grammar -> Props -> Anti-Metaphor -> Staging"
    )
    literal_staging_atmosphere: str = Field(
        ...,
        min_length=1,
        description="Physical lighting/setting implied strictly by the words"
    )
    entities: List[LiteralEntity] = Field(
        ...,
        min_length=1,
        description="List of entities extracted from idiom. Use null placeholder if truly nothing found."
    )
    actions: List[LiteralAction] = Field(
        ...,
        min_length=1,
        description="List of actions/verbs/prepositions. Use null placeholder if truly nothing found."
    )
    relationships: List[LiteralRelationship] = Field(
        default_factory=list,
        description="Relationships linking entities via actions. Can be empty if no relationships exist."
    )


class LiteralExtractionResult(BaseModel):
    """Root model for literal extraction output."""
    
    literal_track: LiteralTrack = Field(
        ...,
        description="The complete literal scene graph"
    )
