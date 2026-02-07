"""
Imageability Checklist Questions for Phase 0 Evaluation.

Each question tests one aspect of how easily an idiom evokes a mental image.
All questions target the LITERAL wording only (no figurative meaning needed).
"""

# System prompt for all imageability questions
IMAGEABILITY_SYSTEM_PROMPT = """
You are an expert Psycholinguist specializing in Dual Coding Theory and Mental Imagery. 

YOUR TASK:
Evaluate the "Imageability" of an idiom based strictly on its LITERAL wording.

CRITICAL RULE - THE "LITERAL WALL":
You must completely IGNORE the figurative or idiomatic meaning of the phrase. 
Pretend you are a robot or a painter who understands English vocabulary but has NO knowledge of cultural idioms. 
- If the idiom is "spill the beans", you MUST imagine actual beans falling out of a container. Do NOT think about "secrets".
- If the idiom is "a little bird told me", you MUST imagine a small bird chirping. Do NOT think about "gossip".

EVALUATION PROTOCOL:
1. Read the idiom text.
2. Visualize the scene described by the words literally.
3. Answer the user's Yes/No question based ONLY on this literal visualization.

Output format: Return ONLY "yes" or "no".
"""

# 14 Imageability Questions
# Format: (question_id, question_template)
# The idiom will be inserted where {idiom} appears
IMAGEABILITY_QUESTIONS = [
    (
        "concrete_nouns",
        'Does the phrase contain any nouns that refer to physical, touchable objects in the real world (ignoring whether they are metaphors)?'
    ),
    (
        "physical_action",
        'Does "{idiom}" include a clear physical action verb?'
    ),
    (
        "single_scene",
        'Can you picture "{idiom}" as one snapshot rather than multiple steps?'
    ),
    (
        "limited_participants",
        'Does the literal scene of "{idiom}" require three or fewer entities?'
    ),
    (
        "everyday_objects",
        'Are the objects in "{idiom}" common in everyday life?'
    ),
    (
        "spatial_layout",
        'Is the spatial layout of "{idiom}" easy to visualize?'
    ),
    (
        "temporal_simplicity",
        'Does "{idiom}" capture a moment, not an extended process?'
    ),
    (
        "dynamic_movement",
        'Does something visibly move or change state in "{idiom}"?'
    ),
    (
        "sensory_details",
        'Is at least one sense (sight/sound/touch) easy to imagine in "{idiom}"?'
    ),
    (
        "real_world",
        'Is the literal scene described by these words physically possible in the real world (e.g., excluding fantasy or physics violations)?'
    ),
    (
        "culture_neutral",
        'Can you visualize "{idiom}" without culture-specific symbols?'
    ),
    (
        "concrete_words",
        'Are the key words in "{idiom}" concrete rather than abstract?'
    ),
    (
        "short_phrase",
        'Is "{idiom}" five words or fewer?'
    ),
    (
        "literal_usage",
        'Can "{idiom}" be used literally without sounding nonsensical?'
    ),
]

def get_imageability_prompts(idiom: str) -> list:
    """
    Generate all 14 imageability question prompts for a given idiom.
    
    Args:
        idiom: The idiom text to evaluate.
        
    Returns:
        List of tuples: (question_id, formatted_question)
    """
    return [
        (qid, question.format(idiom=idiom))
        for qid, question in IMAGEABILITY_QUESTIONS
    ]
