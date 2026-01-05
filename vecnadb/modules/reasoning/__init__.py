"""
VecnaDB Reasoning Module

Provides both deterministic (graph-based) and probabilistic (vector-based)
reasoning capabilities.

Graph Reasoning (AUTHORITATIVE):
- Infers facts from structure
- Validates constraints
- Detects contradictions
- Confidence = 1.0 (deterministic)

Vector Reasoning (ADVISORY):
- Suggests similar entities
- Recommends relations
- Infers types from clustering
- Confidence < 1.0 (probabilistic)

Main Components:
- GraphReasoner: Deterministic structural reasoning
- VectorReasoner: Probabilistic semantic reasoning
- ReasoningEngine: Orchestrates both approaches
"""

from vecnadb.modules.reasoning.GraphReasoner import (
    GraphReasoner,
    InferredRelation,
    ContradictionResult,
    ReasoningResult,
    InferenceType,
    infer_all_relations,
    check_consistency,
)

from vecnadb.modules.reasoning.VectorReasoner import (
    VectorReasoner,
    EntitySuggestion,
    RelationSuggestion,
    TypeSuggestion,
    VectorReasoningResult,
    SuggestionType,
    find_similar,
    suggest_relation_targets,
)

from vecnadb.modules.reasoning.ReasoningEngine import (
    ReasoningEngine,
    CombinedReasoningResult,
    ReasoningMode,
    ReasoningStrategy,
    infer_and_suggest,
    validate_entities,
)

__all__ = [
    # Graph Reasoning
    "GraphReasoner",
    "InferredRelation",
    "ContradictionResult",
    "ReasoningResult",
    "InferenceType",
    "infer_all_relations",
    "check_consistency",
    # Vector Reasoning
    "VectorReasoner",
    "EntitySuggestion",
    "RelationSuggestion",
    "TypeSuggestion",
    "VectorReasoningResult",
    "SuggestionType",
    "find_similar",
    "suggest_relation_targets",
    # Combined Reasoning
    "ReasoningEngine",
    "CombinedReasoningResult",
    "ReasoningMode",
    "ReasoningStrategy",
    "infer_and_suggest",
    "validate_entities",
]
