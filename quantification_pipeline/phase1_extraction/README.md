# Phase 1 Extraction Pipeline

## Overview

This module extracts visual elements from English idioms using LLM-based analysis. It produces two parallel tracks:

1. **Literal Track**: Morphological visualization based on explicit words
2. **Figurative Track**: Visual knowledge graph based on metaphorical meaning

## Project Structure

```
T2IMVI/
├── quantification_pipeline/
│   └── phase1_extraction/
│       ├── main.py                 # Pipeline entry point
│       ├── schemas/                # Pydantic models for output validation
│       │   ├── literal_schema.py
│       │   └── figurative_schema.py
│       ├── models/                 # LLM model abstraction layer
│       │   ├── base_model.py       # Abstract interface
│       │   └── qwen_model.py       # Qwen implementation
│       ├── extractors/             # Extraction logic
│       │   ├── base_extractor.py
│       │   ├── literal_extractor.py
│       │   └── figurative_extractor.py
│       └── utils/                  # Utilities
│           ├── checkpoint.py       # Resume capability
│           └── data_loader.py      # Data loading
└── data/
    └── output/
        └── phase1_extraction/
            ├── output/             # Extraction results
            │   ├── literal/
            │   └── figurative/
            └── checkpoints/        # Progress tracking
```

## Usage

### Basic Usage (Test Mode)

Process 1 idiom for testing:

```bash
# Activate virtual environment first
conda activate T2IMVI  # or your venv

# Run test mode
python main.py --test
```

### Full Extraction

```bash
# Run both tracks on all idioms
python main.py --mode both

# Run only literal track
python main.py --mode literal

# Run only figurative track  
python main.py --mode figurative
```

### Advanced Options

```bash
# Limit to first 10 idioms
python main.py --limit 10

# Process specific idiom IDs
python main.py --ids 1 2 3

# Reset checkpoint and start fresh
python main.py --reset

# Use different model
python main.py --model Qwen/Qwen3-1.7B

# Specify device
python main.py --device cuda:0
```

## Checkpoint & Resume

The pipeline automatically saves progress. If interrupted, simply re-run the same command to resume from where it left off.

Checkpoint files are stored in `D:/Opencode/T2IMVI/data/output/phase1_extraction/checkpoints/`:
- `literal_checkpoint.json`
- `figurative_checkpoint.json`

To start fresh, use `--reset` flag.

## Swapping Models

The model layer is abstracted. To use a different model:

1. Create a new model class inheriting from `BaseExtractionModel`
2. Implement `load_model()`, `generate()`, and `generate_structured()` 
3. Replace model instantiation in `main.py`

Example for using a different model:

```python
from phase1_extraction.models import ModelConfig

config = ModelConfig(
    model_name="Your-Model",
    model_path="org/your-model",
    device="cuda",
    max_new_tokens=4096,
)
```

## Output Format

### Literal Track Output

```json
{
  "idiom_id": 3,
  "idiom": "add fuel to the fire",
  "literal_track": {
    "entities": [
      {"id": "le_1", "content": "fuel", "type": "text_based"},
      {"id": "le_2", "content": "fire", "type": "text_based"},
      {"id": "le_3", "content": "Someone", "type": "placeholder"}
    ],
    "actions": [
      {"id": "la_1", "content": "add"}
    ],
    "relationships": [
      {"subject_id": "le_3", "action_id": "la_1", "object_id": "le_1"}
    ]
  }
}
```

### Figurative Track Output

```json
{
  "idiom_id": 3,
  "idiom": "add fuel to the fire",
  "definition": "...",
  "figurative_track": {
    "thought_process": "1. Core Concept: Escalation...",
    "core_abstract_concept": "Escalation of conflict",
    "abstract_atmosphere": "Warm, aggressive, fiery tones",
    "entities": [...],
    "actions": [...],
    "relationships": [...]
  }
}
```

## Dependencies

```
transformers>=4.40.0
torch>=2.0.0
pydantic>=2.0.0
tqdm>=4.65.0
```

Install with:
```bash
pip install transformers torch pydantic tqdm
```
