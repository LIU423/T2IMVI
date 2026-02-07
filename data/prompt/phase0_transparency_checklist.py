"""
Transparency Checklist Questions for Phase 0 Evaluation.

Each question tests one aspect of semantic transparency - how easily
a learner can guess the figurative meaning from the literal words.
These questions consider BOTH the idiom AND its figurative meaning.
"""

# System prompt for all transparency questions
TRANSPARENCY_SYSTEM_PROMPT = """You evaluate idioms. Answer only "yes" or "no"."""

# 15 Transparency Questions
# Format: (question_id, question_template)
# {idiom} = the idiom text, {meaning} = the figurative meaning
TRANSPARENCY_QUESTIONS = [
    (
        "literal_hints_figurative",
        'Does imagining "{idiom}" literally hint at its meaning: "{meaning}"?'
    ),
    (
        "everyday_word_sense",
        'Do the words in "{idiom}" keep their everyday modern meanings?'
    ),
    (
        "logical_composition",
        'Does simple cause-and-effect link "{idiom}" to "{meaning}"?'
    ),
    (
        "synonym_robust",
        'If a core word in "{idiom}" is replaced by a synonym, is "{meaning}" still guessable?'
    ),
    (
        "paraphrase_from_parts",
        'Can you paraphrase "{meaning}" using only words from "{idiom}"?'
    ),
    (
        "first_time_guessable",
        'Would a first-time learner guess "{meaning}" from "{idiom}"?'
    ),
    (
        "imageable_literal",
        'Is the literal reading of "{idiom}" easy to picture as a concrete scene?'
    ),
    (
        "causal_mapping",
        'Does the literal cause-effect in "{idiom}" match the figurative cause-effect?'
    ),
    (
        "common_vocabulary",
        'Are the words in "{idiom}" common and not culture-specific?'
    ),
    (
        "literal_sentence_possible",
        'Can "{idiom}" plausibly appear as a fully literal sentence?'
    ),
    (
        "morphological_flexibility",
        'Do tense/number changes in "{idiom}" preserve the figurative meaning?'
    ),
    (
        "syntactic_flexibility",
        'Does "{idiom}" convey "{meaning}" in passive or question form?'
    ),
    (
        "no_hidden_metaphors",
        'Are individual words in "{idiom}" literal, with metaphor only at phrase level?'
    ),
    (
        "translation_transparent",
        'Would word-for-word translation of "{idiom}" give a clue to "{meaning}"?'
    ),
    (
        "low_cultural_dependence",
        'Can "{meaning}" be inferred without specific cultural or historical knowledge?'
    ),
]

def get_transparency_prompts(idiom: str, meaning: str) -> list:
    """
    Generate all 15 transparency question prompts for a given idiom.
    
    Args:
        idiom: The idiom text to evaluate.
        meaning: The figurative meaning of the idiom.
        
    Returns:
        List of tuples: (question_id, formatted_question)
    """
    return [
        (qid, question.format(idiom=idiom, meaning=meaning))
        for qid, question in TRANSPARENCY_QUESTIONS
    ]
