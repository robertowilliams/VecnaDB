"""
VecnaDB Vector Reasoner

This module implements probabilistic reasoning using vector embeddings.
Vector reasoning is ADVISORY - it suggests possibilities, never asserts truth.

Key Features:
- Semantic similarity ranking
- Analogical reasoning (A:B :: C:?)
- Cluster-based type inference
- Relation suggestion (advisory only)
- Semantic expansion
- Confidence scoring

Vector reasoning SUGGESTS. Only graph reasoning ASSERTS.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import (
    KnowledgeEntity,
    EmbeddingType,
)
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema


class SuggestionType(str, Enum):
    """Types of vector-based suggestions"""
    SIMILAR_ENTITY = "similar_entity"  # Semantically similar entities
    ANALOGICAL = "analogical"  # Analogical reasoning (A:B :: C:D)
    RELATION_CANDIDATE = "relation_candidate"  # Suggested relation
    TYPE_INFERENCE = "type_inference"  # Suggested entity type
    SEMANTIC_EXPANSION = "semantic_expansion"  # Semantically related concepts


class EntitySuggestion(BaseModel):
    """A suggested entity from vector reasoning"""
    entity: KnowledgeEntity
    confidence: float  # 0.0-1.0, based on similarity
    suggestion_type: SuggestionType
    similarity_score: float
    explanation: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationSuggestion(BaseModel):
    """
    A suggested relation from vector reasoning.

    IMPORTANT: This is ADVISORY only. Vector reasoning cannot assert truth.
    Suggested relations must be validated by graph reasoning or human review.
    """
    source_id: UUID
    relation_type: str
    target_id: UUID
    confidence: float  # Based on vector similarity
    explanation: str
    supporting_evidence: List[str]  # Why this relation is suggested
    advisory_only: bool = True  # Always True for vector suggestions


class TypeSuggestion(BaseModel):
    """Suggested entity type based on embedding clustering"""
    entity_id: UUID
    suggested_type: str
    confidence: float
    similar_entities: List[UUID]  # Entities of this type with similar embeddings
    explanation: str


class VectorReasoningResult(BaseModel):
    """Result of vector reasoning operation"""
    entity_suggestions: List[EntitySuggestion]
    relation_suggestions: List[RelationSuggestion]
    type_suggestions: List[TypeSuggestion]
    reasoning_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorReasoner:
    """
    Probabilistic reasoning using vector embeddings.

    Vector reasoning is ADVISORY:
    - Suggests possibilities based on similarity
    - Confidence scores are probabilistic (0.0-1.0)
    - NEVER asserts truth - only graph reasoning can do that
    - Useful for discovery, exploration, recommendation

    Process:
    1. Embed query or entity
    2. Find semantically similar entities
    3. Analyze patterns in embedding space
    4. Suggest relations based on proximity
    5. Cluster for type inference
    6. Return ranked suggestions with confidence
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema,
        embedding_service: Optional[Any] = None
    ):
        """
        Initialize vector reasoner.

        Args:
            storage: VecnaDB storage interface
            ontology: Ontology schema for type information
            embedding_service: Service to compute embeddings
        """
        self.storage = storage
        self.ontology = ontology
        self.embedding_service = embedding_service

    async def find_similar_entities(
        self,
        entity: KnowledgeEntity,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        entity_types: Optional[List[str]] = None
    ) -> VectorReasoningResult:
        """
        Find semantically similar entities.

        Args:
            entity: Reference entity
            top_k: Number of suggestions
            similarity_threshold: Minimum similarity
            entity_types: Optional type filter

        Returns:
            VectorReasoningResult with similar entity suggestions
        """
        start_time = time.time()

        suggestions = []

        # Get entity's primary embedding
        if not entity.embeddings:
            return VectorReasoningResult(
                entity_suggestions=[],
                relation_suggestions=[],
                type_suggestions=[],
                reasoning_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": "Entity has no embeddings"}
            )

        # Use content embedding
        content_embeddings = [
            emb for emb in entity.embeddings
            if emb.embedding_type == EmbeddingType.CONTENT
        ]

        if not content_embeddings:
            # Fall back to first embedding
            query_vector = entity.embeddings[0].vector
        else:
            query_vector = content_embeddings[0].vector

        # Vector search for similar entities
        results = await self.storage.vector_search(
            query_vector=query_vector,
            entity_types=entity_types,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Convert to suggestions
        for similar_entity, similarity_score in results:
            # Skip the entity itself
            if similar_entity.id == entity.id:
                continue

            suggestion = EntitySuggestion(
                entity=similar_entity,
                confidence=similarity_score,
                suggestion_type=SuggestionType.SIMILAR_ENTITY,
                similarity_score=similarity_score,
                explanation=(
                    f"Semantically similar to '{entity.properties.get('name', entity.id)}' "
                    f"(similarity: {similarity_score:.2f})"
                ),
                metadata={
                    "reference_entity_id": str(entity.id),
                    "entity_type": similar_entity.ontology_type
                }
            )
            suggestions.append(suggestion)

        result = VectorReasoningResult(
            entity_suggestions=suggestions,
            relation_suggestions=[],
            type_suggestions=[],
            reasoning_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "query_entity_id": str(entity.id),
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
        )

        return result

    async def suggest_relations(
        self,
        source_entity: KnowledgeEntity,
        relation_type: str,
        top_k: int = 5
    ) -> VectorReasoningResult:
        """
        Suggest potential target entities for a relation.

        ADVISORY ONLY: These are suggestions based on semantic similarity,
        NOT assertions of truth. Must be validated by graph reasoning.

        Args:
            source_entity: Source entity
            relation_type: Type of relation to suggest
            top_k: Number of suggestions

        Returns:
            VectorReasoningResult with relation suggestions
        """
        start_time = time.time()

        relation_suggestions = []

        # Check if relation type exists in ontology
        if relation_type not in self.ontology.relation_types:
            return VectorReasoningResult(
                entity_suggestions=[],
                relation_suggestions=[],
                type_suggestions=[],
                reasoning_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": f"Unknown relation type: {relation_type}"}
            )

        rel_type_def = self.ontology.relation_types[relation_type]

        # Get allowed target types
        allowed_target_types = rel_type_def.allowed_target_types

        # Find semantically similar entities of the allowed types
        similar_result = await self.find_similar_entities(
            entity=source_entity,
            top_k=top_k * 3,  # Get more candidates
            entity_types=allowed_target_types
        )

        # Convert to relation suggestions
        for entity_suggestion in similar_result.entity_suggestions[:top_k]:
            relation_suggestion = RelationSuggestion(
                source_id=source_entity.id,
                relation_type=relation_type,
                target_id=entity_suggestion.entity.id,
                confidence=entity_suggestion.confidence * 0.8,  # Reduce confidence
                explanation=(
                    f"Suggested {relation_type} target based on semantic similarity. "
                    f"Target '{entity_suggestion.entity.properties.get('name', entity_suggestion.entity.id)}' "
                    f"is semantically similar (score: {entity_suggestion.similarity_score:.2f})"
                ),
                supporting_evidence=[
                    f"Semantic similarity: {entity_suggestion.similarity_score:.2f}",
                    f"Target type '{entity_suggestion.entity.ontology_type}' is allowed",
                ],
                advisory_only=True
            )
            relation_suggestions.append(relation_suggestion)

        result = VectorReasoningResult(
            entity_suggestions=[],
            relation_suggestions=relation_suggestions,
            type_suggestions=[],
            reasoning_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "source_entity_id": str(source_entity.id),
                "relation_type": relation_type,
                "top_k": top_k
            }
        )

        return result

    async def analogical_reasoning(
        self,
        entity_a: KnowledgeEntity,
        entity_b: KnowledgeEntity,
        entity_c: KnowledgeEntity,
        top_k: int = 5
    ) -> VectorReasoningResult:
        """
        Analogical reasoning: A:B :: C:?

        Find entity D such that the relationship between C and D is similar
        to the relationship between A and B.

        Uses vector arithmetic: D ≈ C + (B - A)

        Args:
            entity_a: First entity in analogy
            entity_b: Second entity in analogy
            entity_c: Query entity
            top_k: Number of suggestions

        Returns:
            VectorReasoningResult with analogical suggestions
        """
        start_time = time.time()

        # Get embeddings
        vec_a = self._get_primary_embedding(entity_a)
        vec_b = self._get_primary_embedding(entity_b)
        vec_c = self._get_primary_embedding(entity_c)

        if vec_a is None or vec_b is None or vec_c is None:
            return VectorReasoningResult(
                entity_suggestions=[],
                relation_suggestions=[],
                type_suggestions=[],
                reasoning_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": "Missing embeddings for analogical reasoning"}
            )

        # Vector arithmetic: D = C + (B - A)
        vec_a_np = np.array(vec_a)
        vec_b_np = np.array(vec_b)
        vec_c_np = np.array(vec_c)

        vec_d = vec_c_np + (vec_b_np - vec_a_np)
        vec_d_list = vec_d.tolist()

        # Search for entities similar to the computed vector
        results = await self.storage.vector_search(
            query_vector=vec_d_list,
            entity_types=None,  # Any type
            top_k=top_k + 1,  # +1 in case entity_c appears
            similarity_threshold=0.5
        )

        # Convert to suggestions
        suggestions = []
        for entity_d, similarity_score in results:
            # Skip entity_c itself
            if entity_d.id == entity_c.id:
                continue

            suggestion = EntitySuggestion(
                entity=entity_d,
                confidence=similarity_score * 0.7,  # Reduce confidence for analogy
                suggestion_type=SuggestionType.ANALOGICAL,
                similarity_score=similarity_score,
                explanation=(
                    f"Analogical reasoning: '{entity_a.properties.get('name', entity_a.id)}' "
                    f"is to '{entity_b.properties.get('name', entity_b.id)}' "
                    f"as '{entity_c.properties.get('name', entity_c.id)}' "
                    f"is to '{entity_d.properties.get('name', entity_d.id)}' "
                    f"(similarity: {similarity_score:.2f})"
                ),
                metadata={
                    "analogy_source_a": str(entity_a.id),
                    "analogy_source_b": str(entity_b.id),
                    "analogy_query_c": str(entity_c.id),
                }
            )
            suggestions.append(suggestion)

            if len(suggestions) >= top_k:
                break

        result = VectorReasoningResult(
            entity_suggestions=suggestions,
            relation_suggestions=[],
            type_suggestions=[],
            reasoning_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "reasoning_type": "analogical",
                "analogy": f"{entity_a.id}:{entity_b.id}::{entity_c.id}:?"
            }
        )

        return result

    async def infer_entity_type(
        self,
        entity: KnowledgeEntity,
        top_k: int = 3
    ) -> VectorReasoningResult:
        """
        Suggest entity types based on embedding similarity to known entities.

        ADVISORY ONLY: Type suggestions must be validated before changing
        an entity's type.

        Args:
            entity: Entity to infer type for
            top_k: Number of type suggestions

        Returns:
            VectorReasoningResult with type suggestions
        """
        start_time = time.time()

        # Find similar entities across all types
        similar_result = await self.find_similar_entities(
            entity=entity,
            top_k=50,  # Get many candidates
            similarity_threshold=0.6
        )

        # Group by entity type
        type_groups: Dict[str, List[Tuple[KnowledgeEntity, float]]] = {}
        for suggestion in similar_result.entity_suggestions:
            entity_type = suggestion.entity.ontology_type
            if entity_type not in type_groups:
                type_groups[entity_type] = []
            type_groups[entity_type].append((suggestion.entity, suggestion.similarity_score))

        # Calculate average similarity for each type
        type_scores = []
        for entity_type, entities in type_groups.items():
            avg_similarity = sum(score for _, score in entities) / len(entities)
            type_scores.append((entity_type, avg_similarity, entities))

        # Sort by average similarity
        type_scores.sort(key=lambda x: x[1], reverse=True)

        # Create type suggestions
        type_suggestions = []
        for entity_type, avg_similarity, similar_entities in type_scores[:top_k]:
            suggestion = TypeSuggestion(
                entity_id=entity.id,
                suggested_type=entity_type,
                confidence=avg_similarity,
                similar_entities=[e.id for e, _ in similar_entities[:5]],
                explanation=(
                    f"Entity embeddings are similar to {len(similar_entities)} "
                    f"entities of type '{entity_type}' "
                    f"(avg similarity: {avg_similarity:.2f})"
                )
            )
            type_suggestions.append(suggestion)

        result = VectorReasoningResult(
            entity_suggestions=[],
            relation_suggestions=[],
            type_suggestions=type_suggestions,
            reasoning_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "entity_id": str(entity.id),
                "current_type": entity.ontology_type,
                "top_k": top_k
            }
        )

        return result

    async def semantic_expansion(
        self,
        query_text: str,
        top_k: int = 10,
        entity_types: Optional[List[str]] = None
    ) -> VectorReasoningResult:
        """
        Expand a text query to find semantically related entities.

        Args:
            query_text: Text query
            top_k: Number of suggestions
            entity_types: Optional type filter

        Returns:
            VectorReasoningResult with expanded entities
        """
        start_time = time.time()

        if not self.embedding_service:
            return VectorReasoningResult(
                entity_suggestions=[],
                relation_suggestions=[],
                type_suggestions=[],
                reasoning_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": "Embedding service required for semantic expansion"}
            )

        # Embed query text
        query_vector = await self.embedding_service.embed(query_text)

        # Search for similar entities
        results = await self.storage.vector_search(
            query_vector=query_vector,
            entity_types=entity_types,
            top_k=top_k,
            similarity_threshold=0.5
        )

        # Convert to suggestions
        suggestions = []
        for entity, similarity_score in results:
            suggestion = EntitySuggestion(
                entity=entity,
                confidence=similarity_score,
                suggestion_type=SuggestionType.SEMANTIC_EXPANSION,
                similarity_score=similarity_score,
                explanation=(
                    f"Semantically related to query '{query_text}' "
                    f"(similarity: {similarity_score:.2f})"
                ),
                metadata={
                    "query_text": query_text,
                    "entity_type": entity.ontology_type
                }
            )
            suggestions.append(suggestion)

        result = VectorReasoningResult(
            entity_suggestions=suggestions,
            relation_suggestions=[],
            type_suggestions=[],
            reasoning_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "query_text": query_text,
                "top_k": top_k
            }
        )

        return result

    def _get_primary_embedding(self, entity: KnowledgeEntity) -> Optional[List[float]]:
        """
        Get primary embedding vector for an entity.

        Prioritizes CONTENT embeddings.

        Args:
            entity: Entity to get embedding from

        Returns:
            Embedding vector or None
        """
        if not entity.embeddings:
            return None

        # Prefer CONTENT embeddings
        content_embeddings = [
            emb for emb in entity.embeddings
            if emb.embedding_type == EmbeddingType.CONTENT
        ]

        if content_embeddings:
            return content_embeddings[0].vector

        # Fall back to first embedding
        return entity.embeddings[0].vector


# Convenience functions
async def find_similar(
    entity: KnowledgeEntity,
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    top_k: int = 10
) -> VectorReasoningResult:
    """
    Find similar entities using vector reasoning.

    Args:
        entity: Reference entity
        storage: Storage interface
        ontology: Ontology schema
        top_k: Number of suggestions

    Returns:
        VectorReasoningResult with suggestions
    """
    reasoner = VectorReasoner(storage, ontology)
    return await reasoner.find_similar_entities(entity, top_k)


async def suggest_relation_targets(
    source_entity: KnowledgeEntity,
    relation_type: str,
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    top_k: int = 5
) -> VectorReasoningResult:
    """
    Suggest potential targets for a relation.

    ADVISORY ONLY.

    Args:
        source_entity: Source entity
        relation_type: Relation type
        storage: Storage interface
        ontology: Ontology schema
        top_k: Number of suggestions

    Returns:
        VectorReasoningResult with relation suggestions
    """
    reasoner = VectorReasoner(storage, ontology)
    return await reasoner.suggest_relations(source_entity, relation_type, top_k)
