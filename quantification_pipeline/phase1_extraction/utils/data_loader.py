"""
Data Loader - Load idiom data from JSON file.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_idioms(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load idioms from JSON file.
    
    Expected format:
    [
        {
            "idiom_id": 1,
            "idiom": "a little bird told me",
            "definition": "['...']"
        },
        ...
    ]
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of idiom dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Idiom file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        idioms = json.load(f)
    
    logger.info(f"Loaded {len(idioms)} idioms from {file_path}")
    
    # Deduplicate by idiom_id (keep first occurrence)
    seen_ids = set()
    unique_idioms = []
    
    for idiom in idioms:
        idiom_id = idiom.get("idiom_id")
        if idiom_id not in seen_ids:
            seen_ids.add(idiom_id)
            unique_idioms.append(idiom)
    
    if len(unique_idioms) < len(idioms):
        logger.warning(
            f"Removed {len(idioms) - len(unique_idioms)} duplicate idiom IDs. "
            f"Unique count: {len(unique_idioms)}"
        )
    
    return unique_idioms


def load_prompt_template(file_path: str | Path) -> str:
    """
    Load prompt template from text file.
    
    Args:
        file_path: Path to prompt template file
        
    Returns:
        Prompt template string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    logger.info(f"Loaded prompt template from {file_path}")
    
    return template
