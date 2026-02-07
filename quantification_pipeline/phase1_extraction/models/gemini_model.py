"""
Gemini Model Implementation

Supports Google Gemini API for text generation.
Compatible with the BaseExtractionModel interface.

Features:
- Native JSON mode (responseMimeType + responseJsonSchema)
- Progressive retry strategy with error feedback
- Raw response logging for debugging
- Fault-tolerant JSON extraction
- Stop sequences support
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Type, Optional, Dict, Any, Tuple

from pydantic import BaseModel, ValidationError

from models.base_model import BaseExtractionModel, ModelConfig

logger = logging.getLogger(__name__)


class RawResponseLogger:
    """
    Logs raw LLM responses for debugging failed extractions.
    
    Saves responses to: {debug_dir}/raw_responses/{track_type}/idiom_{id}_attempt_{n}.txt
    """
    
    def __init__(self, debug_dir: Optional[Path] = None):
        """
        Initialize raw response logger.
        
        Args:
            debug_dir: Base directory for debug output. 
                       If None, uses default from project config.
        """
        if debug_dir is None:
            # Try to get from project config
            try:
                import sys
                from pathlib import Path as P
                # Navigate to project root
                _this_dir = P(__file__).parent.resolve()
                _project_root = _this_dir.parent.parent.parent.parent
                if str(_project_root) not in sys.path:
                    sys.path.insert(0, str(_project_root))
                from project_config import OUTPUT_PHASE1_EXTRACTION_DIR
                debug_dir = OUTPUT_PHASE1_EXTRACTION_DIR / "debug"
            except ImportError:
                debug_dir = Path("./debug")
        
        self.debug_dir = Path(debug_dir)
        self.raw_responses_dir = self.debug_dir / "raw_responses"
        self.raw_responses_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"RawResponseLogger initialized. Output: {self.raw_responses_dir}")
    
    def save_raw_response(
        self,
        idiom_id: int,
        track_type: str,
        attempt: int,
        prompt: str,
        raw_response: str,
        error: Optional[str] = None,
    ) -> Path:
        """
        Save a raw response for debugging.
        
        Args:
            idiom_id: The idiom ID being processed
            track_type: "literal" or "figurative"
            attempt: Attempt number (1, 2, or 3)
            prompt: The prompt sent to the model
            raw_response: The raw response from the model
            error: Optional error message if parsing failed
            
        Returns:
            Path to the saved file
        """
        # Create track-specific directory
        track_dir = self.raw_responses_dir / track_type
        track_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"idiom_{idiom_id}_attempt_{attempt}_{timestamp}.txt"
        filepath = track_dir / filename
        
        # Build content
        content = [
            f"=" * 80,
            f"IDIOM ID: {idiom_id}",
            f"TRACK TYPE: {track_type}",
            f"ATTEMPT: {attempt}",
            f"TIMESTAMP: {datetime.now().isoformat()}",
            f"=" * 80,
            "",
            "--- PROMPT ---",
            prompt,
            "",
            "--- RAW RESPONSE ---",
            raw_response if raw_response else "(EMPTY RESPONSE)",
            "",
        ]
        
        if error:
            content.extend([
                "--- ERROR ---",
                error,
                "",
            ])
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        logger.debug(f"Saved raw response to: {filepath}")
        return filepath


class GeminiModel(BaseExtractionModel):
    """
    Gemini API model implementation with robust JSON extraction.
    
    Features:
    - Google Gemini API integration
    - Native JSON mode (responseMimeType + responseJsonSchema) 
    - Progressive retry strategy with error feedback
    - Fault-tolerant JSON extraction
    - Raw response logging for debugging
    - Stop sequences support (<END_JSON>)
    - Compatible with BaseExtractionModel interface
    """
    
    DEFAULT_MODEL = "gemini-2.0-flash"
    
    # Stop sequence to mark end of JSON output
    JSON_END_MARKER = "<END_JSON>"
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        debug_dir: Optional[Path] = None,
        use_native_json_mode: bool = True,
    ):
        """
        Initialize Gemini model.
        
        Args:
            config: Model configuration
            debug_dir: Directory for debug output (raw responses)
            use_native_json_mode: Whether to use Gemini's native JSON mode
        """
        if config is None:
            config = ModelConfig(
                model_name=self.DEFAULT_MODEL,
            )
        super().__init__(config)
        self.client = None
        self.use_native_json_mode = use_native_json_mode
        self.raw_logger = RawResponseLogger(debug_dir)
        
        # Track current extraction context for logging
        self._current_idiom_id: Optional[int] = None
        self._current_track_type: str = "unknown"
        
    def set_extraction_context(self, idiom_id: int, track_type: str) -> None:
        """Set the current extraction context for raw response logging."""
        self._current_idiom_id = idiom_id
        self._current_track_type = track_type
        
    def load_model(self) -> None:
        """Initialize Gemini API client."""
        if self._is_loaded:
            logger.info("Gemini client already initialized, skipping...")
            return
            
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for Gemini API. "
                "Install with: pip install google-genai"
            ) from e
        
        # Get API key from config
        api_key = self.config.extra_kwargs.get("api_key")
        if not api_key:
            raise ValueError(
                "API key is required for Gemini API. "
                "Provide it via ModelConfig.extra_kwargs['api_key']"
            )
        
        logger.info(f"Initializing Gemini client with model: {self.config.model_name}")
        
        # Initialize client with API key
        self.client = genai.Client(api_key=api_key)
        
        self._is_loaded = True
        logger.info(f"Gemini client initialized successfully: {self.config.model_name}")
    
    def generate(
        self,
        prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        use_json_mode: bool = False,
    ) -> str:
        """
        Generate raw text response using Gemini API.
        
        Args:
            prompt: The prompt to send
            json_schema: Optional JSON schema for structured output
            use_json_mode: Whether to force JSON output mode
            
        Returns:
            Generated text response
        """
        if not self._is_loaded:
            self.load_model()
        
        model_name = self.config.model_path or self.config.model_name or self.DEFAULT_MODEL
        
        # Build generation config
        generation_config: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_new_tokens,
        }
        
        # Add stop sequences
        generation_config["stop_sequences"] = [self.JSON_END_MARKER]
        
        # Enable native JSON mode if requested
        if use_json_mode and self.use_native_json_mode:
            generation_config["response_mime_type"] = "application/json"
            if json_schema:
                generation_config["response_schema"] = json_schema
        
        logger.debug(f"Generating with model: {model_name}")
        logger.debug(f"Generation config: {generation_config}")
        
        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config,
            )
            
            # Extract text from response
            if response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini API")
                return ""
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Generate and validate structured JSON output.
        
        Uses progressive retry strategy:
        - Attempt 1: Normal prompt with JSON mode
        - Attempt 2: Feed back error + first 200 chars of raw response
        - Attempt 3: Shorter, stricter schema prompt
        """
        if not self._is_loaded:
            self.load_model()
        
        # Convert Pydantic schema to JSON schema for Gemini
        json_schema = self._pydantic_to_json_schema(response_schema)
        
        last_error: Optional[Exception] = None
        last_raw_response: str = ""
        error_type: str = ""
        current_prompt: str = prompt
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Build prompt based on attempt number
                current_prompt = self._build_progressive_prompt(
                    original_prompt=prompt,
                    attempt=attempt,
                    last_error=error_type,
                    last_raw_response=last_raw_response,
                    schema=response_schema,
                )
                
                # Generate response (use native JSON mode for attempt 1)
                use_json_mode = (attempt == 1) and self.use_native_json_mode
                raw_response = self.generate(
                    current_prompt,
                    json_schema=json_schema if use_json_mode else None,
                    use_json_mode=use_json_mode,
                )
                last_raw_response = raw_response
                
                logger.debug(f"Raw response (attempt {attempt}):\n{raw_response[:500] if raw_response else '(empty)'}...")
                
                # Save raw response for debugging
                if self._current_idiom_id is not None:
                    self.raw_logger.save_raw_response(
                        idiom_id=self._current_idiom_id,
                        track_type=self._current_track_type,
                        attempt=attempt,
                        prompt=current_prompt,
                        raw_response=raw_response,
                    )
                
                # Extract JSON from response
                json_str, extraction_error = self._extract_json_robust(raw_response)
                
                if extraction_error:
                    raise json.JSONDecodeError(extraction_error, raw_response if raw_response else "", 0)
                
                # Parse and validate
                data = json.loads(json_str)
                result = response_schema.model_validate(data)
                
                logger.info(f"Successfully parsed response on attempt {attempt}")
                return result
                
            except json.JSONDecodeError as e:
                last_error = e
                error_type = self._classify_json_error(str(e), last_raw_response)
                logger.warning(f"Parse attempt {attempt} failed ({error_type}): {e}")
                
                # Save error response for debugging
                if self._current_idiom_id is not None:
                    self.raw_logger.save_raw_response(
                        idiom_id=self._current_idiom_id,
                        track_type=self._current_track_type,
                        attempt=attempt,
                        prompt=current_prompt,
                        raw_response=last_raw_response,
                        error=f"{error_type}: {str(e)}",
                    )
                    
            except ValidationError as e:
                last_error = e
                error_type = "schema_validation_failed"
                logger.warning(f"Validation attempt {attempt} failed: {e}")
                
                # Save error response for debugging
                if self._current_idiom_id is not None:
                    self.raw_logger.save_raw_response(
                        idiom_id=self._current_idiom_id,
                        track_type=self._current_track_type,
                        attempt=attempt,
                        prompt=current_prompt,
                        raw_response=last_raw_response,
                        error=f"{error_type}: {str(e)}",
                    )
        
        raise ValueError(
            f"Failed to generate valid structured output after {self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _build_progressive_prompt(
        self,
        original_prompt: str,
        attempt: int,
        last_error: str,
        last_raw_response: str,
        schema: Type[BaseModel],
    ) -> str:
        """
        Build prompt based on attempt number (progressive retry strategy).
        
        Attempt 1: Normal prompt with JSON prefix
        Attempt 2: Error feedback + first 200 chars of raw response
        Attempt 3: Stricter schema prompt with null/[] fallbacks
        """
        # Add JSON prefix to force format
        json_prefix = (
            "IMPORTANT: You MUST respond with ONLY a valid JSON object. "
            "No markdown, no explanations, no text before or after the JSON. "
            f"End your response with {self.JSON_END_MARKER}\n\n"
        )
        
        if attempt == 1:
            # Attempt 1: Normal prompt with JSON prefix
            return json_prefix + original_prompt + f"\n\n{self.JSON_END_MARKER}"
        
        elif attempt == 2:
            # Attempt 2: Error feedback with context
            error_feedback = self._get_error_feedback(last_error, last_raw_response)
            return (
                f"[RETRY - PREVIOUS ATTEMPT FAILED]\n"
                f"{error_feedback}\n\n"
                f"CRITICAL: Output ONLY valid JSON. No markdown code blocks, no explanations.\n\n"
                f"{original_prompt}\n\n"
                f"Remember: ONLY JSON output, nothing else. End with {self.JSON_END_MARKER}"
            )
        
        else:
            # Attempt 3: Stricter prompt with schema summary and null fallbacks
            schema_summary = self._get_schema_summary(schema)
            return (
                f"[FINAL RETRY - USE STRICT FORMAT]\n\n"
                f"You MUST output a JSON object with this EXACT structure:\n"
                f"{schema_summary}\n\n"
                f"RULES:\n"
                f"1. Output ONLY the JSON object - no other text\n"
                f"2. For missing fields, use null for optional values or [] for empty arrays\n"
                f"3. Do NOT use markdown code blocks\n"
                f"4. Ensure all brackets and braces are properly closed\n\n"
                f"Input: {self._extract_input_from_prompt(original_prompt)}\n\n"
                f"JSON output:"
            )
    
    def _get_error_feedback(self, error_type: str, raw_response: str) -> str:
        """Generate error feedback for retry prompt."""
        preview = raw_response[:200] if raw_response else "(empty)"
        
        feedback_map = {
            "empty_response": (
                "ERROR: Your previous response was EMPTY. "
                "You must output a complete JSON object."
            ),
            "markdown_wrapped": (
                f"ERROR: Your previous response was wrapped in markdown code blocks. "
                f"Output raw JSON only, no ```json markers.\n"
                f"Your output started with: {preview}"
            ),
            "missing_closing_bracket": (
                f"ERROR: Your previous JSON was incomplete - missing closing brackets. "
                f"Ensure all {{ have matching }} and all [ have matching ].\n"
                f"Your output started with: {preview}"
            ),
            "invalid_json_syntax": (
                f"ERROR: Your previous response had invalid JSON syntax. "
                f"Check for trailing commas, unquoted strings, or special characters.\n"
                f"Your output started with: {preview}"
            ),
            "not_json": (
                f"ERROR: Your previous response was not JSON at all. "
                f"Output ONLY a JSON object starting with {{ and ending with }}.\n"
                f"Your output started with: {preview}"
            ),
        }
        
        return feedback_map.get(error_type, f"ERROR: Previous attempt failed. Raw output: {preview}")
    
    def _classify_json_error(self, error_msg: str, raw_response: str) -> str:
        """Classify the type of JSON parsing error."""
        if not raw_response or raw_response.strip() == "":
            return "empty_response"
        
        raw_stripped = raw_response.strip()
        
        # Check for markdown wrapping
        if raw_stripped.startswith("```"):
            return "markdown_wrapped"
        
        # Check for missing closing brackets
        open_braces = raw_stripped.count('{') 
        close_braces = raw_stripped.count('}')
        open_brackets = raw_stripped.count('[')
        close_brackets = raw_stripped.count(']')
        
        if open_braces > close_braces or open_brackets > close_brackets:
            return "missing_closing_bracket"
        
        # Check if it looks like JSON at all
        if not (raw_stripped.startswith('{') or raw_stripped.startswith('[')):
            return "not_json"
        
        return "invalid_json_syntax"
    
    def _get_schema_summary(self, schema: Type[BaseModel]) -> str:
        """Generate a concise schema summary for strict retry prompt."""
        try:
            # Get the JSON schema
            json_schema = schema.model_json_schema()
            
            # Extract key structure
            if "properties" in json_schema:
                props = json_schema["properties"]
                lines = ["{"]
                for key, value in props.items():
                    type_str = value.get("type", "object")
                    lines.append(f'  "{key}": <{type_str}>,')
                lines.append("}")
                return "\n".join(lines)
            
            return json.dumps(json_schema, indent=2)[:500]
        except Exception:
            return '{"literal_track": {"entities": [...], "actions": [...], "relationships": [...]}}'
    
    def _extract_input_from_prompt(self, prompt: str) -> str:
        """Extract the input (e.g., idiom) from the original prompt."""
        # Look for idiom pattern
        match = re.search(r'idiom:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: return first 100 chars
        return prompt[:100] + "..."
    
    def _extract_json_robust(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Robust JSON extraction with multiple fallback strategies.
        
        Returns:
            Tuple of (extracted_json, error_message)
            If successful, error_message is None
        """
        if not text or text.strip() == "":
            return "", "Empty response from model"
        
        text = text.strip()
        
        # Remove end marker if present
        if self.JSON_END_MARKER in text:
            text = text.split(self.JSON_END_MARKER)[0].strip()
        
        # Strategy 1: Try to find JSON in markdown code fence
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            candidate = json_match.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate, None
        
        # Strategy 2: Find outermost balanced braces
        json_str = self._extract_balanced_json(text)
        if json_str and self._is_valid_json(json_str):
            return json_str, None
        
        # Strategy 3: Try to repair common issues
        repaired = self._try_repair_json(text)
        if repaired and self._is_valid_json(repaired):
            return repaired, None
        
        # Strategy 4: Extract any JSON-like substring
        json_like = self._extract_json_substring(text)
        if json_like and self._is_valid_json(json_like):
            return json_like, None
        
        # All strategies failed
        return text, "Could not extract valid JSON from response"
    
    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract JSON using balanced brace matching."""
        # Find first { 
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # Track braces and brackets
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
                
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]
        
        # Unbalanced - return what we have
        return text[start_idx:] if start_idx >= 0 else None
    
    def _try_repair_json(self, text: str) -> Optional[str]:
        """Try to repair common JSON issues."""
        # Remove leading/trailing non-JSON content
        text = text.strip()
        
        # Remove markdown markers
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        # Find JSON boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start == -1 or end == -1 or end <= start:
            return None
        
        candidate = text[start:end + 1]
        
        # Try to fix common issues
        # 1. Remove trailing commas before } or ]
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        
        # 2. Add missing closing braces/brackets
        open_braces = candidate.count('{')
        close_braces = candidate.count('}')
        if open_braces > close_braces:
            candidate += '}' * (open_braces - close_braces)
        
        open_brackets = candidate.count('[')
        close_brackets = candidate.count(']')
        if open_brackets > close_brackets:
            candidate += ']' * (open_brackets - close_brackets)
        
        return candidate
    
    def _extract_json_substring(self, text: str) -> Optional[str]:
        """Extract any valid JSON object substring."""
        # Try each { as a potential start
        start = 0
        while True:
            start = text.find('{', start)
            if start == -1:
                break
            
            # Try each } as a potential end
            end = start + 1
            while True:
                end = text.find('}', end)
                if end == -1:
                    break
                
                candidate = text[start:end + 1]
                if self._is_valid_json(candidate):
                    return candidate
                end += 1
            
            start += 1
        
        return None
    
    def _is_valid_json(self, text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    
    def _pydantic_to_json_schema(self, schema: Type[BaseModel]) -> Dict[str, Any]:
        """Convert Pydantic model to JSON schema for Gemini."""
        try:
            return schema.model_json_schema()
        except Exception as e:
            logger.warning(f"Failed to generate JSON schema: {e}")
            return {}
    
    def unload(self) -> None:
        """Clean up Gemini client."""
        if self.client is not None:
            self.client = None
        self._is_loaded = False
        logger.info("Gemini client unloaded")
