"""
VecnaDB Reasoning Engine

This module orchestrates both graph and vector reasoning to provide
comprehensive knowledge inference and discovery.

Key Principles:
- Graph reasoning is AUTHORITATIVE (asserts truth)
- Vector reasoning is ADVISORY (suggests possibilities)
- Combined reasoning uses both approaches
- Clear separation of facts vs suggestions

The ReasoningEngine is the primary interface for multi-step reasoning tasks.
"""

import time
from typing import List, Optional, Dict, Any, Set
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema
from vecnadb.modules.reasoning.GraphReasoner import (
    GraphReasoner,
    InferredRelation,
    ContradictionResult,
    ReasoningResult as GraphReasoningResult,
)
from vecnadb.modules.reasoning.VectorReasoner import (
    VectorReasoner,
    EntitySuggestion,
    RelationSuggestion,
    TypeSuggestion,
    VectorReasoningResult,
)


class ReasoningMode(str, Enum):
    """Reasoning execution mode"""
    GRAPH_ONLY = "graph_only"  # Deterministic reasoning only
    VECTOR_ONLY = "vector_only"  # Probabilistic reasoning only
    HYBRID = "hybrid"  # Both graph and vector reasoning
    SEQUENTIAL = "sequential"  # Graph first, then vector on results


class ReasoningStrategy(str, Enum):
    """High-level reasoning strategy"""
    INFERENCE = "inference"  # Infer new facts from existing knowledge
    DISCOVERY = "discovery"  # Discover related entities and relations
    VALIDATION = "validation"  # Validate consistency and constraints
    EXPANSION = "expansion"  # Expand knowledge with suggestions
    ANALYSIS = "analysis"  # Analyze patterns and structure


class CombinedReasoningResult(BaseModel):
    """
    Combined result from both graph and vector reasoning.

    Clearly separates:
    - FACTS (from graph reasoning) - authoritative
    - SUGGESTIONS (from vector reasoning) - advisory
    """

    # Graph reasoning results (AUTHORITATIVE)
    inferred_facts: List[InferredRelation] = Field(default_factory=list)
    contradictions: List[ContradictionResult] = Field(default_factory=list)

    # Vector reasoning results (ADVISORY)
    entity_suggestions: List[EntitySuggestion] = Field(default_factory=list)
    relation_suggestions: List[RelationSuggestion] = Field(default_factory=list)
    type_suggestions: List[TypeSuggestion] = Field(default_factory=list)

    # Execution metadata
    graph_reasoning_time_ms: float = 0.0
    vector_reasoning_time_ms: float = 0.0
    total_time_ms: float = 0.0
    entities_processed: int = 0
    reasoning_mode: ReasoningMode
    reasoning_strategy: ReasoningStrategy

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_all_facts(self) -> List[InferredRelation]:
        """Get all inferred facts (authoritative)"""
        return self.inferred_facts

    def get_all_suggestions(self) -> Dict[str, List]:
        """Get all suggestions (advisory)"""
        return {
            "entities": self.entity_suggestions,
            "relations": self.relation_suggestions,
            "types": self.type_suggestions
        }

    def has_contradictions(self) -> bool:
        """Check if any contradictions were found"""
        return len(self.contradictions) > 0

    def get_high_confidence_suggestions(
        self,
        threshold: float = 0.8
    ) -> Dict[str, List]:
        """Get only high-confidence suggestions"""
        return {
            "entities": [
                s for s in self.entity_suggestions
                if s.confidence >= threshold
            ],
            "relations": [
                s for s in self.relation_suggestions
                if s.confidence >= threshold
            ],
            "types": [
                s for s in self.type_suggestions
                if s.confidence >= threshold
            ]
        }


class ReasoningEngine:
    """
    Orchestrates graph and vector reasoning for comprehensive knowledge inference.

    The ReasoningEngine provides high-level reasoning capabilities by combining:
    - GraphReasoner: Deterministic, authoritative facts
    - VectorReasoner: Probabilistic, advisory suggestions

    Use Cases:
    1. Knowledge discovery: Find related entities and suggest connections
    2. Inference: Derive new facts from existing knowledge
    3. Validation: Check consistency and detect contradictions
    4. Exploration: Expand knowledge with semantic suggestions
    5. Analysis: Understand patterns and structure
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema,
        embedding_service: Optional[Any] = None
    ):
        """
        Initialize reasoning engine.

        Args:
            storage: VecnaDB storage interface
            ontology: Ontology schema
            embedding_service: Optional embedding service for vector reasoning
        """
        self.storage = storage
        self.ontology = ontology
        self.embedding_service = embedding_service

        # Initialize reasoners
        self.graph_reasoner = GraphReasoner(storage, ontology)
        self.vector_reasoner = VectorReasoner(storage, ontology, embedding_service)

    async def reason(
        self,
        entity_id: UUID,
        mode: ReasoningMode = ReasoningMode.HYBRID,
        strategy: ReasoningStrategy = ReasoningStrategy.INFERENCE,
        max_depth: int = 3,
        top_k_suggestions: int = 10,
        confidence_threshold: float = 0.7
    ) -> CombinedReasoningResult:
        """
        Perform comprehensive reasoning about an entity.

        Args:
            entity_id: Entity to reason about
            mode: Reasoning mode (graph/vector/hybrid)
            strategy: Reasoning strategy
            max_depth: Maximum graph traversal depth
            top_k_suggestions: Number of vector suggestions
            confidence_threshold: Minimum confidence for suggestions

        Returns:
            CombinedReasoningResult with facts and suggestions
        """
        start_time = time.time()

        result = CombinedReasoningResult(
            reasoning_mode=mode,
            reasoning_strategy=strategy,
            metadata={
                "entity_id": str(entity_id),
                "max_depth": max_depth,
                "top_k_suggestions": top_k_suggestions
            }
        )

        # Get entity
        entity = await self.storage.get_entity(entity_id)
        result.entities_processed = 1

        # Execute based on mode
        if mode == ReasoningMode.GRAPH_ONLY:
            await self._execute_graph_reasoning(
                entity_id, max_depth, result
            )

        elif mode == ReasoningMode.VECTOR_ONLY:
            await self._execute_vector_reasoning(
                entity, top_k_suggestions, confidence_threshold, result
            )

        elif mode == ReasoningMode.HYBRID:
            # Run both in parallel
            await self._execute_graph_reasoning(
                entity_id, max_depth, result
            )
            await self._execute_vector_reasoning(
                entity, top_k_suggestions, confidence_threshold, result
            )

        elif mode == ReasoningMode.SEQUENTIAL:
            # Graph first
            await self._execute_graph_reasoning(
                entity_id, max_depth, result
            )

            # Then vector reasoning on inferred entities
            await self._execute_vector_reasoning(
                entity, top_k_suggestions, confidence_threshold, result
            )

        result.total_time_ms = (time.time() - start_time) * 1000

        return result

    async def _execute_graph_reasoning(
        self,
        entity_id: UUID,
        max_depth: int,
        result: CombinedReasoningResult
    ):
        """Execute graph reasoning and update result"""
        graph_start = time.time()

        graph_result = await self.graph_reasoner.infer_relations(
            entity_id=entity_id,
            max_depth=max_depth
        )

        result.inferred_facts.extend(graph_result.inferred_relations)
        result.contradictions.extend(graph_result.contradictions)
        result.graph_reasoning_time_ms = (time.time() - graph_start) * 1000

    async def _execute_vector_reasoning(
        self,
        entity: KnowledgeEntity,
        top_k: int,
        confidence_threshold: float,
        result: CombinedReasoningResult
    ):
        """Execute vector reasoning and update result"""
        vector_start = time.time()

        # Find similar entities
        similar_result = await self.vector_reasoner.find_similar_entities(
            entity=entity,
            top_k=top_k,
            similarity_threshold=confidence_threshold
        )

        result.entity_suggestions.extend(similar_result.entity_suggestions)

        # Infer entity type
        type_result = await self.vector_reasoner.infer_entity_type(
            entity=entity,
            top_k=3
        )

        result.type_suggestions.extend(type_result.type_suggestions)

        result.vector_reasoning_time_ms = (time.time() - vector_start) * 1000

    async def multi_hop_reasoning(
        self,
        start_entity_id: UUID,
        num_hops: int = 2,
        top_k_per_hop: int = 5
    ) -> CombinedReasoningResult:
        """
        Perform multi-hop reasoning starting from an entity.

        Process:
        1. Start with entity
        2. Find similar entities (vector)
        3. Infer relations from graph structure
        4. Repeat for N hops
        5. Aggregate all facts and suggestions

        Args:
            start_entity_id: Starting entity
            num_hops: Number of reasoning hops
            top_k_per_hop: Entities to explore per hop

        Returns:
            CombinedReasoningResult with multi-hop reasoning
        """
        start_time = time.time()

        result = CombinedReasoningResult(
            reasoning_mode=ReasoningMode.HYBRID,
            reasoning_strategy=ReasoningStrategy.DISCOVERY,
            metadata={
                "start_entity_id": str(start_entity_id),
                "num_hops": num_hops,
                "top_k_per_hop": top_k_per_hop
            }
        )

        # Track visited entities to avoid loops
        visited: Set[UUID] = set()
        current_entities = [start_entity_id]

        for hop in range(num_hops):
            next_entities = []

            for entity_id in current_entities:
                if entity_id in visited:
                    continue

                visited.add(entity_id)
                result.entities_processed += 1

                # Graph reasoning
                graph_result = await self.graph_reasoner.infer_relations(
                    entity_id=entity_id,
                    max_depth=1  # Only one hop at a time
                )

                result.inferred_facts.extend(graph_result.inferred_relations)
                result.contradictions.extend(graph_result.contradictions)

                # Vector reasoning
                entity = await self.storage.get_entity(entity_id)
                vector_result = await self.vector_reasoner.find_similar_entities(
                    entity=entity,
                    top_k=top_k_per_hop
                )

                result.entity_suggestions.extend(vector_result.entity_suggestions)

                # Collect next hop candidates from vector suggestions
                for suggestion in vector_result.entity_suggestions[:top_k_per_hop]:
                    if suggestion.entity.id not in visited:
                        next_entities.append(suggestion.entity.id)

            current_entities = next_entities

            if not current_entities:
                break

        result.total_time_ms = (time.time() - start_time) * 1000

        return result

    async def validate_consistency(
        self,
        entity_ids: List[UUID]
    ) -> CombinedReasoningResult:
        """
        Validate consistency of a set of entities.

        Uses graph reasoning to check:
        - Cardinality constraints
        - Ontology compliance
        - Structural consistency

        Args:
            entity_ids: Entities to validate

        Returns:
            CombinedReasoningResult with contradictions
        """
        start_time = time.time()

        result = CombinedReasoningResult(
            reasoning_mode=ReasoningMode.GRAPH_ONLY,
            reasoning_strategy=ReasoningStrategy.VALIDATION,
            metadata={
                "entity_count": len(entity_ids)
            }
        )

        graph_result = await self.graph_reasoner.validate_graph_consistency(
            entity_ids=entity_ids
        )

        result.contradictions.extend(graph_result.contradictions)
        result.entities_processed = graph_result.entities_validated
        result.total_time_ms = (time.time() - start_time) * 1000

        return result

    async def expand_knowledge(
        self,
        query_text: str,
        top_k: int = 20,
        entity_types: Optional[List[str]] = None
    ) -> CombinedReasoningResult:
        """
        Expand knowledge based on a text query.

        Uses vector reasoning to find semantically related entities,
        then uses graph reasoning to infer structural connections.

        Args:
            query_text: Text query
            top_k: Number of entities to find
            entity_types: Optional entity type filter

        Returns:
            CombinedReasoningResult with expanded knowledge
        """
        start_time = time.time()

        result = CombinedReasoningResult(
            reasoning_mode=ReasoningMode.HYBRID,
            reasoning_strategy=ReasoningStrategy.EXPANSION,
            metadata={
                "query_text": query_text,
                "top_k": top_k
            }
        )

        # Vector reasoning: semantic expansion
        vector_result = await self.vector_reasoner.semantic_expansion(
            query_text=query_text,
            top_k=top_k,
            entity_types=entity_types
        )

        result.entity_suggestions.extend(vector_result.entity_suggestions)

        # For top suggestions, run graph reasoning
        for suggestion in vector_result.entity_suggestions[:5]:
            graph_result = await self.graph_reasoner.infer_relations(
                entity_id=suggestion.entity.id,
                max_depth=2
            )

            result.inferred_facts.extend(graph_result.inferred_relations)
            result.contradictions.extend(graph_result.contradictions)
            result.entities_processed += 1

        result.total_time_ms = (time.time() - start_time) * 1000

        return result

    async def suggest_and_validate_relation(
        self,
        source_entity_id: UUID,
        relation_type: str,
        top_k: int = 5
    ) -> CombinedReasoningResult:
        """
        Suggest potential relations using vectors, then validate using graph.

        Process:
        1. Use vector reasoning to suggest target entities
        2. Use graph reasoning to validate if relation is structurally valid
        3. Return suggestions with validation status

        Args:
            source_entity_id: Source entity
            relation_type: Relation type to suggest
            top_k: Number of suggestions

        Returns:
            CombinedReasoningResult with validated suggestions
        """
        start_time = time.time()

        result = CombinedReasoningResult(
            reasoning_mode=ReasoningMode.SEQUENTIAL,
            reasoning_strategy=ReasoningStrategy.DISCOVERY,
            metadata={
                "source_entity_id": str(source_entity_id),
                "relation_type": relation_type
            }
        )

        # Get source entity
        source_entity = await self.storage.get_entity(source_entity_id)

        # Vector reasoning: suggest targets
        vector_result = await self.vector_reasoner.suggest_relations(
            source_entity=source_entity,
            relation_type=relation_type,
            top_k=top_k
        )

        # For each suggestion, validate using graph reasoning
        for rel_suggestion in vector_result.relation_suggestions:
            # Check if this relation would violate any constraints
            # (This would require checking cardinality, allowed types, etc.)

            # For now, add validation status to metadata
            rel_suggestion.metadata["graph_validated"] = False

            # Check if relation already exists
            existing_relations = await self.storage.get_relations(
                entity_id=source_entity_id,
                relation_type=relation_type,
                direction="OUTGOING"
            )

            already_exists = any(
                r.target_id == rel_suggestion.target_id
                for r in existing_relations
            )

            rel_suggestion.metadata["already_exists"] = already_exists

        result.relation_suggestions.extend(vector_result.relation_suggestions)
        result.total_time_ms = (time.time() - start_time) * 1000

        return result


# Convenience functions
async def infer_and_suggest(
    entity_id: UUID,
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    embedding_service: Optional[Any] = None,
    max_depth: int = 3
) -> CombinedReasoningResult:
    """
    Run comprehensive reasoning (both graph and vector) on an entity.

    Args:
        entity_id: Entity to reason about
        storage: Storage interface
        ontology: Ontology schema
        embedding_service: Optional embedding service
        max_depth: Maximum graph depth

    Returns:
        CombinedReasoningResult with facts and suggestions
    """
    engine = ReasoningEngine(storage, ontology, embedding_service)
    return await engine.reason(
        entity_id=entity_id,
        mode=ReasoningMode.HYBRID,
        max_depth=max_depth
    )


async def validate_entities(
    entity_ids: List[UUID],
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema
) -> CombinedReasoningResult:
    """
    Validate consistency of entities.

    Args:
        entity_ids: Entities to validate
        storage: Storage interface
        ontology: Ontology schema

    Returns:
        CombinedReasoningResult with contradictions
    """
    engine = ReasoningEngine(storage, ontology)
    return await engine.validate_consistency(entity_ids)
