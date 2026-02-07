"""
Figurative Track Schema - Pydantic models for structured output validation.

Based on phase1_figurative_extraction_specialist.txt prompt requirements.
Enforces strict JSON schema compliance for LLM outputs.

Note: entities, actions, and relationships can be empty arrays.
object_id in relationships can be null for intransitive actions.
"""

from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class FigurativeEntity(BaseModel):
    """Entity representing a visual symbol for the figurative meaning."""
    
    id: str = Field(
        ...,
        pattern=r"^fe_\d+$",
        description="Unique identifier in format 'fe_X' where X is a number"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Visual symbol name"
    )
    type: Literal["symbolic", "placeholder", "null"] = Field(
        ...,
        description="symbolic for visual symbols, placeholder for implied agents, null if empty"
    )
    requires_cultural_context: bool = Field(
        ...,
        description="Whether this symbol requires cultural context to understand"
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Why this symbol represents the concept"
    )


class FigurativeAction(BaseModel):
    """Action in the figurative scene."""
    
    id: str = Field(
        ...,
        pattern=r"^fa_\d+$",
        description="Unique identifier in format 'fa_X' where X is a number"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Action description"
    )
    requires_cultural_context: bool = Field(
        ...,
        description="Whether this action requires cultural context"
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Why this action represents the concept"
    )


class FigurativeRelationship(BaseModel):
    """Relationship linking subject, action, and object via IDs."""
    
    subject_id: str = Field(
        ...,
        pattern=r"^fe_\d+$",
        description="Entity ID of the subject"
    )
    action_id: str = Field(
        ...,
        pattern=r"^fa_\d+$",
        description="Action ID reference - MUST match an ID from actions list"
    )
    object_id: Optional[str] = Field(
        None,
        pattern=r"^fe_\d+$",
        description="Entity ID of the object. Can be null for intransitive actions."
    )


class FigurativeTrack(BaseModel):
    """Complete figurative visual knowledge graph for an idiom."""
    
    thought_process: str = Field(
        ...,
        min_length=10,
        description="CoT reasoning: Core Concept -> Brainstorming -> Cultural Check -> Atmosphere"
    )
    core_abstract_concept: str = Field(
        ...,
        min_length=1,
        description="The central abstract theme"
    )
    abstract_atmosphere: str = Field(
        ...,
        min_length=1,
        description="Visual keywords for lighting/color"
    )
    entities: List[FigurativeEntity] = Field(
        default_factory=list,
        description="List of visual symbol entities. Can be empty if no entities found."
    )
    actions: List[FigurativeAction] = Field(
        default_factory=list,
        description="List of actions in the scene. Can be empty if no actions found."
    )
    relationships: List[FigurativeRelationship] = Field(
        default_factory=list,
        description="Relationships linking entities via actions. Can be empty if no relationships found."
    )


class FigurativeExtractionResult(BaseModel):
    """Root model for figurative extraction output."""
    
    figurative_track: FigurativeTrack = Field(
        ...,
        description="The complete figurative visual knowledge graph"
    )
