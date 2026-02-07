"""
IU (Image Understanding) Calculator.

This module implements the IU scoring logic that evaluates whether
an image embodies the core abstract concept through:
1. Relationship-based evaluation (if relationship scores > threshold)
2. Entity-action based evaluation (fallback when relationships invalid)

Score = P("yes") using VQAScore methodology.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple

from PIL import Image

from ..models.iu_base_model import BaseIUModel, YesNoLogitResult
from ..utils.data_handler import ImageInfo, FigurativeData

logger = logging.getLogger(__name__)


@dataclass
class RelationshipValidation:
    """
    Result of validating a single relationship.
    
    Attributes:
        is_valid: True if all components have score > threshold
        subject_content: Content of the subject entity
        action_content: Content of the action
        object_content: Content of the object entity
        subject_score: Score of the subject entity
        action_score: Score of the action
        object_score: Score of the object entity
    """
    is_valid: bool
    subject_content: str = ""
    action_content: str = ""
    object_content: str = ""
    subject_score: float = 0.0
    action_score: float = 0.0
    object_score: float = 0.0


@dataclass
class EntityActionSelection:
    """
    Result of selecting best entity and action for fallback mode.
    
    Attributes:
        has_valid_entity: True if at least one entity has score > threshold
        has_valid_action: True if at least one action has score > threshold
        entity_content: Content of the highest-scoring valid entity
        action_content: Content of the highest-scoring valid action
        entity_score: Score of the selected entity
        action_score: Score of the selected action
    """
    has_valid_entity: bool
    has_valid_action: bool
    entity_content: str = ""
    action_content: str = ""
    entity_score: float = 0.0
    action_score: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        """True if both valid entity and action exist."""
        return self.has_valid_entity and self.has_valid_action


@dataclass
class IUEvaluationResult:
    """
    Complete result of IU evaluation for an image.
    
    Attributes:
        iu_score: The final IU score (P("yes") or 0.0)
        mode: "relationships", "entity_action", or "zero"
        prompt_used: The prompt that was used for evaluation
        details: Additional details about the evaluation
    """
    iu_score: float
    mode: str
    prompt_used: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class IUCalculator:
    """
    Calculator for Image Understanding scores.
    
    Uses a Vision-Language Model to evaluate alignment between
    an image and its core abstract concept.
    
    Scoring Logic:
    1. For each relationship, check if subject_id, action_id, object_id 
       correspond to entities/actions with score > threshold
    2. If ANY relationship is fully valid: Use relationships prompt
    3. If NO relationships are valid:
       a. If NO entities OR NO actions have score > threshold: Return 0.0
       b. Otherwise: Use highest-scoring entity + action with without_relationships prompt
    4. Score = P("yes") from VQAScore methodology
    """
    
    def __init__(
        self,
        model: BaseIUModel,
        relationships_prompt_template: str,
        without_relationships_prompt_template: str,
        score_threshold: float = 0.1,
    ):
        """
        Initialize IU calculator.
        
        Args:
            model: Loaded VLM model implementing BaseIUModel
            relationships_prompt_template: Template from phase2_iu_relationships.txt
            without_relationships_prompt_template: Template from phase2_iu_without_relationships.txt
            score_threshold: Minimum score for entity/action to be valid (default: 0.1)
        """
        self.model = model
        self.relationships_prompt_template = relationships_prompt_template
        self.without_relationships_prompt_template = without_relationships_prompt_template
        self.score_threshold = score_threshold
    
    def _get_entity_by_id(
        self, 
        entity_id: str, 
        entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find entity by ID."""
        for entity in entities:
            if entity.get("id") == entity_id:
                return entity
        return None
    
    def _get_action_by_id(
        self, 
        action_id: str, 
        actions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find action by ID."""
        for action in actions:
            if action.get("id") == action_id:
                return action
        return None
    
    def validate_relationship(
        self,
        relationship: Dict[str, str],
        entities: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
    ) -> RelationshipValidation:
        """
        Validate a single relationship against the score threshold.
        
        A relationship is valid if:
        - subject_id exists in entities with score > threshold
        - action_id exists in actions with score > threshold
        - object_id exists in entities with score > threshold
        
        Args:
            relationship: Dict with subject_id, action_id, object_id
            entities: List of entity dicts with id, content, score
            actions: List of action dicts with id, content, score
            
        Returns:
            RelationshipValidation with validity status and content
        """
        subject_id = relationship.get("subject_id", "")
        action_id = relationship.get("action_id", "")
        object_id = relationship.get("object_id", "")
        
        subject = self._get_entity_by_id(subject_id, entities)
        action = self._get_action_by_id(action_id, actions)
        obj = self._get_entity_by_id(object_id, entities)
        
        # Check if all components exist
        if subject is None or action is None or obj is None:
            return RelationshipValidation(is_valid=False)
        
        subject_score = subject.get("score", 0.0)
        action_score = action.get("score", 0.0)
        object_score = obj.get("score", 0.0)
        
        # Check if all scores > threshold
        is_valid = (
            subject_score > self.score_threshold and
            action_score > self.score_threshold and
            object_score > self.score_threshold
        )
        
        return RelationshipValidation(
            is_valid=is_valid,
            subject_content=subject.get("content", ""),
            action_content=action.get("content", ""),
            object_content=obj.get("content", ""),
            subject_score=subject_score,
            action_score=action_score,
            object_score=object_score,
        )
    
    def find_valid_relationship(
        self,
        figurative_track: Dict[str, Any],
    ) -> Optional[RelationshipValidation]:
        """
        Find the first valid relationship in the figurative track.
        
        Args:
            figurative_track: Dict with entities, actions, relationships
            
        Returns:
            First valid RelationshipValidation, or None if none found
        """
        entities = figurative_track.get("entities", [])
        actions = figurative_track.get("actions", [])
        relationships = figurative_track.get("relationships", [])
        
        for relationship in relationships:
            validation = self.validate_relationship(relationship, entities, actions)
            if validation.is_valid:
                logger.debug(
                    f"Valid relationship found: {validation.subject_content} "
                    f"'{validation.action_content}' {validation.object_content}"
                )
                return validation
        
        return None
    
    def select_best_entity_action(
        self,
        figurative_track: Dict[str, Any],
    ) -> EntityActionSelection:
        """
        Select the highest-scoring entity and action above threshold.
        
        Args:
            figurative_track: Dict with entities, actions
            
        Returns:
            EntityActionSelection with best entity/action if available
        """
        entities = figurative_track.get("entities", [])
        actions = figurative_track.get("actions", [])
        
        # Find highest-scoring entity above threshold
        valid_entities = [
            e for e in entities 
            if e.get("score", 0.0) > self.score_threshold
        ]
        best_entity = max(valid_entities, key=lambda e: e.get("score", 0.0), default=None)
        
        # Find highest-scoring action above threshold
        valid_actions = [
            a for a in actions 
            if a.get("score", 0.0) > self.score_threshold
        ]
        best_action = max(valid_actions, key=lambda a: a.get("score", 0.0), default=None)
        
        has_valid_entity = best_entity is not None
        has_valid_action = best_action is not None
        
        return EntityActionSelection(
            has_valid_entity=has_valid_entity,
            has_valid_action=has_valid_action,
            entity_content=best_entity.get("content", "") if best_entity else "",
            action_content=best_action.get("content", "") if best_action else "",
            entity_score=best_entity.get("score", 0.0) if best_entity else 0.0,
            action_score=best_action.get("score", 0.0) if best_action else 0.0,
        )
    
    def calculate(
        self,
        image: Union[Image.Image, Path, str],
        figurative_data: FigurativeData,
    ) -> IUEvaluationResult:
        """
        Calculate IU score for an image.
        
        Logic:
        1. Check for valid relationships (all components score > threshold)
        2. If valid relationship found: Use relationships prompt -> P("yes")
        3. If no valid relationships:
           a. Check if any entities AND any actions have score > threshold
           b. If not: Return 0.0
           c. If yes: Use best entity + action with without_relationships prompt -> P("yes")
        
        Args:
            image: Image to evaluate
            figurative_data: Loaded figurative data with entities, actions, relationships
            
        Returns:
            IUEvaluationResult with score and evaluation details
        """
        figurative_track = figurative_data.raw_data.get("figurative_track", {})
        core_abstract_concept = figurative_track.get("core_abstract_concept", "")
        
        if not core_abstract_concept:
            logger.warning("No core_abstract_concept found, returning 0.0")
            return IUEvaluationResult(
                iu_score=0.0,
                mode="zero",
                details={"reason": "missing core_abstract_concept"},
            )
        
        # Step 1: Try to find a valid relationship
        valid_relationship = self.find_valid_relationship(figurative_track)
        
        if valid_relationship is not None:
            # Use relationships prompt
            prompt = self.model.format_relationships_prompt(
                core_abstract_concept=core_abstract_concept,
                subject=valid_relationship.subject_content,
                action=valid_relationship.action_content,
                obj=valid_relationship.object_content,
                system_prompt=self.relationships_prompt_template,
            )
            
            result = self.model.get_yes_no_probs(image, prompt)
            
            logger.debug(
                f"Relationships mode: P(yes)={result.yes_prob:.4f}, P(no)={result.no_prob:.4f}"
            )
            
            return IUEvaluationResult(
                iu_score=result.iu_score,
                mode="relationships",
                prompt_used=prompt,
                details={
                    "subject": valid_relationship.subject_content,
                    "action": valid_relationship.action_content,
                    "object": valid_relationship.object_content,
                    "yes_prob": result.yes_prob,
                    "no_prob": result.no_prob,
                },
            )
        
        # Step 2: No valid relationships, try entity-action fallback
        selection = self.select_best_entity_action(figurative_track)
        
        if not selection.is_valid:
            # Either no entities or no actions above threshold -> return 0
            reason = []
            if not selection.has_valid_entity:
                reason.append("no valid entities")
            if not selection.has_valid_action:
                reason.append("no valid actions")
            
            logger.debug(f"Zero mode: {', '.join(reason)}")
            
            return IUEvaluationResult(
                iu_score=0.0,
                mode="zero",
                details={"reason": ", ".join(reason)},
            )
        
        # Use entity-action without relationships prompt
        prompt = self.model.format_without_relationships_prompt(
            core_abstract_concept=core_abstract_concept,
            entity=selection.entity_content,
            action=selection.action_content,
            system_prompt=self.without_relationships_prompt_template,
        )
        
        result = self.model.get_yes_no_probs(image, prompt)
        
        logger.debug(
            f"Entity-action mode: P(yes)={result.yes_prob:.4f}, P(no)={result.no_prob:.4f}"
        )
        
        return IUEvaluationResult(
            iu_score=result.iu_score,
            mode="entity_action",
            prompt_used=prompt,
            details={
                "entity": selection.entity_content,
                "action": selection.action_content,
                "entity_score": selection.entity_score,
                "action_score": selection.action_score,
                "yes_prob": result.yes_prob,
                "no_prob": result.no_prob,
            },
        )
    
    def calculate_for_image_info(
        self,
        image_info: ImageInfo,
        figurative_data: FigurativeData,
        image: Image.Image,
    ) -> float:
        """
        Calculate IU score for an ImageInfo object.
        
        Convenience method that uses pre-loaded data.
        
        Args:
            image_info: ImageInfo with paths (for logging)
            figurative_data: Loaded figurative data
            image: Pre-loaded PIL Image
            
        Returns:
            IU score (P("yes") or 0.0)
        """
        result = self.calculate(image, figurative_data)
        
        logger.info(
            f"{image_info.key}: mode={result.mode}, iu_score={result.iu_score:.4f}"
        )
        
        return result.iu_score
