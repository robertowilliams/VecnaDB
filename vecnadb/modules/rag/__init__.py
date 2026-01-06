"""
VecnaDB RAG (Retrieval-Augmented Generation) Module

Provides ontology-guided RAG with hallucination prevention and complete provenance.

Key Components:
1. OntologyGuidedRAG: Core RAG system with ontology constraints
2. ContextValidator: Validates context quality and compliance
3. HallucinationPrevention: Detects and prevents hallucinations
4. AnswerGrounding: Provides complete provenance tracking

Features:
- Ontology-constrained retrieval
- Context validation
- Hallucination detection
- Complete provenance
- Citation generation
- Answer verification

Principles:
- All context must be ontology-valid
- All claims must be grounded in knowledge graph
- Full traceability from answer to sources
- Confidence scoring for trustworthiness
"""

from vecnadb.modules.rag.OntologyGuidedRAG import (
    OntologyGuidedRAG,
    RAGContext,
    RAGAnswer,
    ContextItem,
    GroundedClaim,
    QueryIntent,
    ContextSource,
    ask,
    retrieve,
)

from vecnadb.modules.rag.ContextValidator import (
    ContextValidator,
    ValidationResult,
    ContextValidationReport,
    ValidationLevel,
    ValidationIssue,
    validate_rag_context,
    filter_invalid_context,
)

from vecnadb.modules.rag.HallucinationPrevention import (
    HallucinationPrevention,
    HallucinationReport,
    HallucinationDetection,
    HallucinationType,
    Claim,
    check_hallucination,
    verify_answer,
)

from vecnadb.modules.rag.AnswerGrounding import (
    AnswerGrounding,
    GroundedAnswer,
    ProvenanceGraph,
    ProvenanceLink,
    Citation,
    CitationStyle,
    ProvenanceType,
    ground_answer,
    trace_claim,
)

__all__ = [
    # Core RAG
    "OntologyGuidedRAG",
    "RAGContext",
    "RAGAnswer",
    "ContextItem",
    "GroundedClaim",
    "QueryIntent",
    "ContextSource",
    "ask",
    "retrieve",
    # Context Validation
    "ContextValidator",
    "ValidationResult",
    "ContextValidationReport",
    "ValidationLevel",
    "ValidationIssue",
    "validate_rag_context",
    "filter_invalid_context",
    # Hallucination Prevention
    "HallucinationPrevention",
    "HallucinationReport",
    "HallucinationDetection",
    "HallucinationType",
    "Claim",
    "check_hallucination",
    "verify_answer",
    # Answer Grounding
    "AnswerGrounding",
    "GroundedAnswer",
    "ProvenanceGraph",
    "ProvenanceLink",
    "Citation",
    "CitationStyle",
    "ProvenanceType",
    "ground_answer",
    "trace_claim",
]
