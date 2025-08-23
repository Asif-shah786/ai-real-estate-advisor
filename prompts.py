"""
AI Real Estate Advisor - Prompt Management System

This file contains all the prompts used throughout the application, organized by function
and with clear documentation for easy modification and improvement.

Author: AI Assistant
Version: 1.0
Last Updated: 2025-08-23
"""

# =============================================================================
# QUERY REWRITING PROMPTS
# =============================================================================

QUERY_REWRITING_PROMPT = """
Rewrite the following user query, correcting spelling and grammar while keeping 
the original intent intact: {query}

Instructions:
- Correct spelling and grammar
- Preserve real estate and Manchester-specific details
- Retain the original meaning
- Your ONLY task is to rewrite — do not answer

"""

# =============================================================================
# CONTEXTUALIZATION PROMPTS (History-Aware Retriever)
# =============================================================================

CONTEXTUALIZATION_SYSTEM_PROMPT = """
You are a query rewriter that resolves references in real estate conversations.

Rules:
1. Preserve references to previous properties or areas (e.g., “last one”, “first property”).
2. Expand shorthand into explicit details using prior conversation context (address + listing ID if available).
3. If the question is standalone, leave it unchanged.
4. If context is missing or unclear, output a clarifying rephrasing such as “Which property do you mean?”.
5. Always maintain property order and numbering.
6. Do not answer the question; your ONLY task is to rewrite it with full clarity.

Examples:
- “what is the crime rate in last one?” → “What is the crime rate for the last property shown (384 Chester Road, Old Trafford, Manchester M16)?”
- “is there any schools nearby second property?” → “Are there schools near the second property from the previous list (Dinton Street, Manchester M15)?”

"""

# =============================================================================
# MAIN QA CHAIN PROMPTS
# =============================================================================

# Base prompt template for all QA chains
BASE_QA_PROMPT = """
You are a helpful AI real estate advisor for Manchester properties.  

Instructions:
1. Use ONLY the provided context to answer. Do not rely on outside knowledge.
2. Present multiple properties in a structured format (bullet points or table).
3. Always include key details: price, bedrooms, location, property type.
4. If context includes citations (URL, council, source), show them at the end (e.g., “Source: Salford Council”).
5. If the requested information is not present in the context, clearly say so and suggest a clarifying question.
6. Handle follow-ups by correctly mapping “first/second/last property” to the right listing.
7. Maintain property order and numbering.
8. Greater Manchester is treated as Manchester for search queries.

Context:
{context}

Question:
{question}

Answer:
"""

# =============================================================================
# PROMPT CONFIGURATIONS
# =============================================================================

# Configuration for different prompt versions
PROMPT_CONFIGS = {
    "lcel": {
        "name": "LCEL Version with Context Handling",
        "description": "Used for LCEL-based history-aware retriever chains",
        "prompt": BASE_QA_PROMPT,
        "use_case": "Advanced conversational chains with memory",
    },
    "standard": {
        "name": "Standard Version with Context Handling",
        "description": "Used for standard ConversationalRetrievalChain",
        "prompt": BASE_QA_PROMPT,
        "use_case": "Standard conversational chains",
    },
    "fallback": {
        "name": "Fallback Version with Context Handling",
        "description": "Used when other chain configurations fail",
        "prompt": BASE_QA_PROMPT,
        "use_case": "Emergency fallback for error handling",
    },
}

# =============================================================================
# PROMPT UTILITY FUNCTIONS
# =============================================================================


def get_prompt_config(version: str) -> dict:
    """
    Get prompt configuration for a specific version.

    Args:
        version (str): Prompt version ('lcel', 'standard', 'fallback')

    Returns:
        dict: Prompt configuration with name, description, prompt, and use_case

    Raises:
        ValueError: If version is not supported
    """
    if version not in PROMPT_CONFIGS:
        supported_versions = list(PROMPT_CONFIGS.keys())
        raise ValueError(
            f"Unsupported prompt version '{version}'. Supported versions: {supported_versions}"
        )

    return PROMPT_CONFIGS[version]


def list_available_prompts() -> list:
    """
    List all available prompt configurations.

    Returns:
        list: List of available prompt versions
    """
    return list(PROMPT_CONFIGS.keys())


def get_prompt_template(version: str) -> str:
    """
    Get the prompt template string for a specific version.

    Args:
        version (str): Prompt version ('lcel', 'standard', 'fallback')

    Returns:
        str: Prompt template string
    """
    config = get_prompt_config(version)
    return config["prompt"]


# =============================================================================
# PROMPT IMPROVEMENT GUIDELINES
# =============================================================================

PROMPT_IMPROVEMENT_GUIDELINES = """
GUIDELINES FOR IMPROVING PROMPTS:

1. CONTEXT PRESERVATION:
   - Always test contextual questions like "second property", "last one"
   - Ensure property order and numbering is maintained
   - Verify that follow-up questions reference the correct properties

2. RESPONSE QUALITY:
   - Test with various property types (flats, houses, studios)
   - Verify crime data, school data, and transport data retrieval
   - Ensure responses are structured and informative

3. ERROR HANDLING:
   - Test with unclear or ambiguous questions
   - Verify fallback responses when context is missing
   - Ensure graceful handling of edge cases

4. PERFORMANCE:
   - Keep prompts concise but comprehensive
   - Avoid redundant instructions
   - Test response generation speed

5. TESTING SCENARIOS:
   - "show me 3 properties in manchester"
   - "what is the crime rate in second property?"
   - "are there schools near the first property?"
   - "tell me about the last property shown"
"""

# =============================================================================
# VERSION HISTORY
# =============================================================================

VERSION_HISTORY = {
    "1.0": {
        "date": "2025-08-23",
        "changes": [
            "Initial prompt organization",
            "Enhanced contextualization for property references",
            "Added critical instructions for property ordering",
            "Unified prompt structure across all versions",
        ],
    }
}

if __name__ == "__main__":
    # Demo usage
    print("Available Prompt Versions:")
    for version in list_available_prompts():
        config = get_prompt_config(version)
        print(f"\n{config['name']}")
        print(f"Description: {config['description']}")
        print(f"Use Case: {config['use_case']}")

    print(f"\nPrompt Improvement Guidelines:")
    print(PROMPT_IMPROVEMENT_GUIDELINES)
