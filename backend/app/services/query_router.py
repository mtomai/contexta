"""
Query Router Module

Intelligently routes queries to avoid unnecessary LLM calls.
Classifies queries and determines optimal handling strategy.
"""

import re
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.config import get_settings

settings = get_settings()


class QueryType(Enum):
    """Types of query handling strategies."""
    DIRECT_ANSWER = "direct"       # Can answer directly from chunks (no LLM needed)
    SYNTHESIS = "synthesis"        # Requires LLM synthesis
    FULL_DOCUMENT = "full_document"  # Requires full document context
    OUT_OF_SCOPE = "out_of_scope"  # Query not answerable from documents
    CLARIFICATION = "clarification"  # Query too vague, needs clarification


# Keywords that trigger FULL_DOCUMENT mode
FULL_DOCUMENT_KEYWORDS = [
    "riassumi tutto", "intero documento", "tutto il file",
    "leggi tutto", "sintesi completa", "riassunto generale",
    "spiegami tutto"
]

@dataclass
class RoutingDecision:
    """Result of query routing decision."""
    query_type: QueryType
    direct_answer: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""


# Patterns for simple lookup queries
SIMPLE_QUERY_PATTERNS = [
    (r"^(?:cos['\s]?[èe]|che\s*cos['\s]?[èe])\s+(.+?)\??$", "definition"),
    (r"^definisci\s+(.+?)\.?$", "definition"),
    (r"^(?:chi\s+[èe]|chi\s+era)\s+(.+?)\??$", "entity"),
    (r"^(?:quando|in\s+che\s+anno)\s+(.+?)\??$", "date"),
    (r"^(?:dove|in\s+quale\s+luogo)\s+(.+?)\??$", "location"),
]

# Keywords that suggest complex synthesis is needed
SYNTHESIS_KEYWORDS = [
    "confronta", "paragona", "differenza", "similitudini",
    "riassumi", "spiega", "analizza", "valuta",
    "perché", "come", "quali sono", "elenca",
    "vantaggi", "svantaggi", "pro e contro",
    "relazione tra", "correlazione"
]


def classify_query(
    query: str, 
    top_results: List[Dict[str, Any]],
    top_score: float
) -> RoutingDecision:
    """
    Classify query and determine handling strategy.
    
    Args:
        query: User query
        top_results: Top retrieved chunks with metadata
        top_score: Similarity score of best match (0-1, higher = more similar)
        
    Returns:
        RoutingDecision with handling strategy
    """
    query_lower = query.lower().strip()

    # 1. If query explicitly asks for full document analysis, route to synthesis regardless of scores
    for kw in FULL_DOCUMENT_KEYWORDS:
        if kw in query_lower:
            return RoutingDecision(
                query_type=QueryType.FULL_DOCUMENT,
                confidence=0.95,
                reason="Richiesta esplicita di analisi dell'intero documento"
            )
    
    # Check if query is out of scope (very low similarity)
    if top_score < settings.low_relevance_threshold:
        return RoutingDecision(
            query_type=QueryType.OUT_OF_SCOPE,
            confidence=1 - top_score,
            reason=f"Similarity score too low ({top_score:.2f})"
        )
    
    # Check if query requires synthesis (complex question)
    if _requires_synthesis(query_lower):
        return RoutingDecision(
            query_type=QueryType.SYNTHESIS,
            confidence=0.9,
            reason="Query requires analysis/synthesis"
        )
    
    # Check for simple lookup with high confidence match
    if top_score >= settings.high_relevance_threshold:
        for pattern, pattern_type in SIMPLE_QUERY_PATTERNS:
            match = re.match(pattern, query_lower, re.IGNORECASE)
            if match:
                # Potentially can answer directly
                direct = _try_extract_direct_answer(
                    query_lower, 
                    match.group(1) if match.groups() else query_lower,
                    top_results[0] if top_results else None,
                    pattern_type
                )
                if direct:
                    return RoutingDecision(
                        query_type=QueryType.DIRECT_ANSWER,
                        direct_answer=direct,
                        confidence=top_score,
                        reason=f"High confidence {pattern_type} lookup"
                    )
    
    # Default: requires LLM synthesis
    return RoutingDecision(
        query_type=QueryType.SYNTHESIS,
        confidence=0.7,
        reason="Standard query requiring LLM"
    )


def _requires_synthesis(query: str) -> bool:
    """Check if query requires complex synthesis."""
    for keyword in SYNTHESIS_KEYWORDS:
        if keyword in query:
            return True
    
    # Multiple question marks suggest multi-part question
    if query.count("?") > 1:
        return True
    
    # Long queries typically need synthesis
    if len(query.split()) > 15:
        return True
    
    return False


def _try_extract_direct_answer(
    query: str,
    subject: str,
    top_chunk: Optional[Dict[str, Any]],
    pattern_type: str
) -> Optional[str]:
    """
    Try to extract a direct answer from top chunk.
    
    Currently conservative - only extracts if chunk starts with definition-like patterns.
    
    Args:
        query: Original query
        subject: Extracted subject from query
        top_chunk: Best matching chunk
        pattern_type: Type of pattern matched
        
    Returns:
        Direct answer string or None if extraction not confident
    """
    if not top_chunk:
        return None
    
    text = top_chunk.get("text", top_chunk.get("chunk_text", ""))
    metadata = top_chunk.get("metadata", {})
    
    # For now, be conservative - only handle definitions with clear patterns
    if pattern_type == "definition":
        subject_clean = subject.strip().lower()
        text_lower = text.lower()
        
        # Check if chunk starts with or contains definition of subject
        definition_patterns = [
            rf"{re.escape(subject_clean)}\s+[èe]\s+",
            rf"{re.escape(subject_clean)}:\s+",
            rf"(?:il|la|lo|l'|un|una)\s+{re.escape(subject_clean)}\s+[èe]\s+",
        ]
        
        for pattern in definition_patterns:
            if re.search(pattern, text_lower):
                # Found definition - extract first sentence or two
                sentences = re.split(r'[.!?]', text)
                if sentences:
                    answer = ". ".join(sentences[:2]).strip()
                    if answer:
                        source = f"[{metadata.get('document_name', 'documento')}, pagina {metadata.get('page', '?')}]"
                        return f"{answer}. {source}"
    
    # Could not extract confidently
    return None


def should_use_llm(decision: RoutingDecision) -> bool:
    """
    Determine if LLM should be called based on routing decision.
    
    Args:
        decision: Routing decision
        
    Returns:
        True if LLM call is needed
    """
    return decision.query_type in [QueryType.SYNTHESIS, QueryType.CLARIFICATION, QueryType.FULL_DOCUMENT]


def get_fallback_response(decision: RoutingDecision) -> Optional[str]:
    """
    Get fallback response when LLM is not needed.
    
    Args:
        decision: Routing decision
        
    Returns:
        Response string or None if LLM is needed
    """
    if decision.query_type == QueryType.OUT_OF_SCOPE:
        return settings.out_of_scope_message
    
    if decision.query_type == QueryType.DIRECT_ANSWER and decision.direct_answer:
        return decision.direct_answer
    
    return None
