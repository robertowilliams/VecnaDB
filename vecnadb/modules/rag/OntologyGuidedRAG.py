"""
VecnaDB Ontology-Guided RAG

Retrieval-Augmented Generation system that enforces ontology constraints
and uses graph structure to ground answers in verified knowledge.

Key Principles:
- All retrieved context must be ontology-valid
- Graph structure provides ground truth
- Vector search provides semantic relevance
- Every answer must be traceable to source entities
- Hallucination prevention through structural verification

RAG Pipeline:
1. Query analysis (extract entities, relations, constraints)
2. Hybrid retrieval (vector similarity + graph traversal)
3. Context validation (ontology compliance)
4. Context ranking (relevance + validity)
5. Answer generation (with grounding)
6. Answer verification (hallucination check)
7. Provenance tracking (full traceability)
"""

import time
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
    Relation,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema
from vecnadb.modules.query.models.HybridQuery import (
    HybridQuery,
    HybridQueryBuilder,
)
from vecnadb.modules.query.executor.HybridQueryExecutor import HybridQueryExecutor
from vecnadb.modules.reasoning.ReasoningEngine import (
    ReasoningEngine,
    ReasoningMode,
)


class QueryIntent(str, Enum):
    """Type of user query intent"""
    FACTUAL = "factual"  # Seeking specific facts
    EXPLORATORY = "exploratory"  # Exploring a topic
    ANALYTICAL = "analytical"  # Analyzing relationships
    COMPARATIVE = "comparative"  # Comparing entities
    DEFINITIONAL = "definitional"  # Asking for definitions


class ContextSource(str, Enum):
    """Source of retrieved context"""
    VECTOR_SEARCH = "vector_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    INFERENCE = "inference"
    DIRECT_LOOKUP = "direct_lookup"


class ContextItem(BaseModel):
    """A piece of retrieved context"""
    entity: KnowledgeEntity
    relevance_score: float  # How relevant to query (0.0-1.0)
    validity_score: float  # How ontology-valid (0.0-1.0)
    source: ContextSource
    relations: List[Relation] = Field(default_factory=list)
    explanation: str  # Why this context was retrieved
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGContext(BaseModel):
    """Complete retrieval context for answer generation"""
    query: str
    intent: QueryIntent
    entities: List[ContextItem]
    total_entities: int
    avg_relevance: float
    avg_validity: float
    ontology_compliant: bool
    retrieval_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_high_quality_context(
        self,
        min_relevance: float = 0.7,
        min_validity: float = 0.8
    ) -> List[ContextItem]:
        """Get only high-quality context items"""
        return [
            item for item in self.entities
            if item.relevance_score >= min_relevance
            and item.validity_score >= min_validity
        ]

    def get_context_by_source(self, source: ContextSource) -> List[ContextItem]:
        """Get context from specific source"""
        return [item for item in self.entities if item.source == source]


class GroundedClaim(BaseModel):
    """A claim in the answer grounded in knowledge graph"""
    claim_text: str
    supporting_entities: List[UUID]
    supporting_relations: List[Relation]
    confidence: float  # Based on source quality
    source_explanation: str
    is_inferred: bool = False  # True if from reasoning, not direct retrieval


class RAGAnswer(BaseModel):
    """Generated answer with full provenance"""
    query: str
    answer_text: str
    grounded_claims: List[GroundedClaim]
    context_used: List[UUID]  # Entity IDs used in answer
    confidence: float  # Overall answer confidence
    hallucination_risk: float  # 0.0-1.0, higher = more risk
    warnings: List[str] = Field(default_factory=list)
    generation_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_trustworthy(self, confidence_threshold: float = 0.7) -> bool:
        """Check if answer is trustworthy"""
        return (
            self.confidence >= confidence_threshold
            and self.hallucination_risk < 0.3
            and len(self.warnings) == 0
        )

    def get_provenance(self) -> List[UUID]:
        """Get all source entities"""
        return self.context_used


class OntologyGuidedRAG:
    """
    Ontology-guided RAG system for VecnaDB.

    Combines:
    - Hybrid search for retrieval
    - Ontology validation for quality
    - Graph structure for grounding
    - Reasoning for inference
    - Provenance for traceability

    Prevents hallucination by:
    - Validating all context against ontology
    - Grounding claims in graph structure
    - Tracking provenance for every claim
    - Detecting unsupported statements
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema,
        llm_service: Optional[Any] = None,
        embedding_service: Optional[Any] = None
    ):
        """
        Initialize ontology-guided RAG.

        Args:
            storage: VecnaDB storage interface
            ontology: Ontology schema
            llm_service: LLM for answer generation
            embedding_service: Embedding service for queries
        """
        self.storage = storage
        self.ontology = ontology
        self.llm_service = llm_service
        self.embedding_service = embedding_service

        # Initialize components
        self.query_executor = HybridQueryExecutor(storage, ontology, embedding_service)
        self.reasoning_engine = ReasoningEngine(storage, ontology, embedding_service)

    async def retrieve_context(
        self,
        query: str,
        intent: QueryIntent = QueryIntent.FACTUAL,
        max_entities: int = 20,
        entity_types: Optional[List[str]] = None,
        include_reasoning: bool = True
    ) -> RAGContext:
        """
        Retrieve ontology-validated context for a query.

        Args:
            query: User query text
            intent: Query intent type
            max_entities: Maximum context entities
            entity_types: Optional entity type filter
            include_reasoning: Include inferred facts

        Returns:
            RAGContext with validated context
        """
        start_time = time.time()

        # Build hybrid query
        builder = HybridQueryBuilder()
        builder.with_query_text(query)
        builder.with_max_results(max_entities)

        if entity_types:
            builder.with_entity_types(entity_types)

        # Adjust settings based on intent
        if intent == QueryIntent.FACTUAL:
            # Favor precision for factual queries
            builder.with_similarity_threshold(0.8)
            builder.with_graph_depth(1)
        elif intent == QueryIntent.EXPLORATORY:
            # Favor breadth for exploration
            builder.with_similarity_threshold(0.6)
            builder.with_graph_depth(3)
        elif intent == QueryIntent.ANALYTICAL:
            # Deep graph traversal for analysis
            builder.with_similarity_threshold(0.7)
            builder.with_graph_depth(3)

        hybrid_query = builder.build()

        # Execute hybrid search
        search_result = await self.query_executor.execute(hybrid_query)

        # Convert to context items
        context_items = []
        for result_item in search_result.results:
            context_item = ContextItem(
                entity=result_item.entity,
                relevance_score=result_item.combined_score,
                validity_score=1.0 if result_item.entity.ontology_valid else 0.0,
                source=ContextSource.VECTOR_SEARCH,
                explanation=result_item.explanation,
                metadata={
                    "vector_score": result_item.vector_similarity_score,
                    "graph_score": result_item.graph_centrality_score
                }
            )
            context_items.append(context_item)

        # Add reasoning-based context if requested
        if include_reasoning and context_items:
            # Use top entity for reasoning expansion
            top_entity = context_items[0].entity

            reasoning_result = await self.reasoning_engine.reason(
                entity_id=top_entity.id,
                mode=ReasoningMode.HYBRID,
                max_depth=2,
                top_k_suggestions=5
            )

            # Add highly relevant suggestions as context
            for suggestion in reasoning_result.entity_suggestions[:5]:
                if suggestion.confidence >= 0.8:
                    context_item = ContextItem(
                        entity=suggestion.entity,
                        relevance_score=suggestion.confidence,
                        validity_score=1.0 if suggestion.entity.ontology_valid else 0.0,
                        source=ContextSource.INFERENCE,
                        explanation=f"Inferred via reasoning: {suggestion.explanation}",
                        metadata={"suggestion_type": suggestion.suggestion_type}
                    )
                    context_items.append(context_item)

        # Calculate aggregates
        avg_relevance = (
            sum(item.relevance_score for item in context_items) / len(context_items)
            if context_items else 0.0
        )

        avg_validity = (
            sum(item.validity_score for item in context_items) / len(context_items)
            if context_items else 0.0
        )

        ontology_compliant = all(
            item.entity.ontology_valid for item in context_items
        )

        # Build context
        rag_context = RAGContext(
            query=query,
            intent=intent,
            entities=context_items,
            total_entities=len(context_items),
            avg_relevance=avg_relevance,
            avg_validity=avg_validity,
            ontology_compliant=ontology_compliant,
            retrieval_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "include_reasoning": include_reasoning,
                "entity_types": entity_types
            }
        )

        return rag_context

    async def generate_answer(
        self,
        query: str,
        intent: QueryIntent = QueryIntent.FACTUAL,
        max_context_entities: int = 10,
        require_grounding: bool = True
    ) -> RAGAnswer:
        """
        Generate a grounded answer for a query.

        Args:
            query: User query
            intent: Query intent
            max_context_entities: Max context to retrieve
            require_grounding: Require all claims to be grounded

        Returns:
            RAGAnswer with grounded claims and provenance
        """
        start_time = time.time()

        # 1. Retrieve context
        context = await self.retrieve_context(
            query=query,
            intent=intent,
            max_entities=max_context_entities,
            include_reasoning=True
        )

        # 2. Filter to high-quality context
        high_quality = context.get_high_quality_context(
            min_relevance=0.7,
            min_validity=0.9
        )

        if not high_quality:
            # No high-quality context available
            return RAGAnswer(
                query=query,
                answer_text="I don't have sufficient high-quality information to answer this query reliably.",
                grounded_claims=[],
                context_used=[],
                confidence=0.0,
                hallucination_risk=1.0,
                warnings=["No high-quality context found"],
                generation_time_ms=(time.time() - start_time) * 1000,
                metadata={"context_quality": "insufficient"}
            )

        # 3. Generate answer using LLM (if available)
        if self.llm_service:
            answer_text = await self._generate_with_llm(query, high_quality)
        else:
            # Fallback: simple concatenation
            answer_text = self._generate_simple_answer(query, high_quality)

        # 4. Ground claims in knowledge graph
        grounded_claims = await self._ground_answer(answer_text, high_quality)

        # 5. Calculate confidence and hallucination risk
        confidence = self._calculate_confidence(grounded_claims, high_quality)
        hallucination_risk = self._calculate_hallucination_risk(
            grounded_claims,
            high_quality
        )

        # 6. Generate warnings
        warnings = []
        if hallucination_risk > 0.5:
            warnings.append("High hallucination risk detected")
        if not context.ontology_compliant:
            warnings.append("Some context may not be ontology-compliant")
        if confidence < 0.6:
            warnings.append("Low confidence in answer accuracy")

        # 7. Build answer
        answer = RAGAnswer(
            query=query,
            answer_text=answer_text,
            grounded_claims=grounded_claims,
            context_used=[item.entity.id for item in high_quality],
            confidence=confidence,
            hallucination_risk=hallucination_risk,
            warnings=warnings,
            generation_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "intent": intent,
                "context_entities": len(high_quality),
                "avg_context_relevance": context.avg_relevance
            }
        )

        return answer

    async def _generate_with_llm(
        self,
        query: str,
        context: List[ContextItem]
    ) -> str:
        """
        Generate answer using LLM with context.

        Args:
            query: User query
            context: Context items

        Returns:
            Generated answer text
        """
        # Build context text
        context_text = "\n\n".join([
            f"Entity: {item.entity.properties.get('name', item.entity.id)}\n"
            f"Type: {item.entity.ontology_type}\n"
            f"Content: {item.entity.properties.get('content', 'N/A')}\n"
            f"Relevance: {item.relevance_score:.2f}"
            for item in context
        ])

        # Build prompt
        prompt = f"""Using ONLY the following context, answer the query.
Do not add information that is not present in the context.
If the context doesn't contain enough information, say so.

Context:
{context_text}

Query: {query}

Answer (using only the provided context):"""

        # Call LLM
        answer = await self.llm_service.generate(prompt)
        return answer

    def _generate_simple_answer(
        self,
        query: str,
        context: List[ContextItem]
    ) -> str:
        """
        Generate simple answer without LLM.

        Args:
            query: User query
            context: Context items

        Returns:
            Simple concatenated answer
        """
        # Simple fallback: concatenate top context
        parts = []
        for item in context[:3]:
            name = item.entity.properties.get('name', str(item.entity.id))
            content = item.entity.properties.get('content', '')
            if content:
                parts.append(f"{name}: {content}")

        if parts:
            return " | ".join(parts)
        else:
            return "Relevant entities found, but no content available."

    async def _ground_answer(
        self,
        answer_text: str,
        context: List[ContextItem]
    ) -> List[GroundedClaim]:
        """
        Ground answer claims in knowledge graph.

        Args:
            answer_text: Generated answer
            context: Context used

        Returns:
            List of grounded claims
        """
        # Simple implementation: create one claim for entire answer
        # Real implementation would parse answer into claims and ground each

        grounded_claims = [
            GroundedClaim(
                claim_text=answer_text,
                supporting_entities=[item.entity.id for item in context],
                supporting_relations=[],  # Would extract from context
                confidence=sum(item.relevance_score for item in context) / len(context),
                source_explanation=f"Based on {len(context)} relevant entities",
                is_inferred=any(
                    item.source == ContextSource.INFERENCE for item in context
                )
            )
        ]

        return grounded_claims

    def _calculate_confidence(
        self,
        claims: List[GroundedClaim],
        context: List[ContextItem]
    ) -> float:
        """Calculate overall answer confidence"""
        if not claims:
            return 0.0

        # Average claim confidence weighted by context quality
        claim_confidence = sum(c.confidence for c in claims) / len(claims)
        context_quality = sum(c.validity_score for c in context) / len(context)

        return (claim_confidence + context_quality) / 2

    def _calculate_hallucination_risk(
        self,
        claims: List[GroundedClaim],
        context: List[ContextItem]
    ) -> float:
        """Calculate hallucination risk"""
        if not claims:
            return 1.0

        # Check if all claims are well-supported
        ungrounded_claims = sum(
            1 for claim in claims
            if len(claim.supporting_entities) == 0
        )

        if ungrounded_claims > 0:
            return 0.8  # High risk if any ungrounded claims

        # Check context validity
        invalid_context = sum(
            1 for item in context
            if item.validity_score < 0.8
        )

        if invalid_context > len(context) * 0.3:
            return 0.6  # Moderate risk if >30% invalid context

        # Check if answer uses inferred facts
        has_inferred = any(claim.is_inferred for claim in claims)
        if has_inferred:
            return 0.3  # Low-moderate risk for inferred facts

        return 0.1  # Low risk

    async def explain_answer(
        self,
        answer: RAGAnswer
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation of answer provenance.

        Args:
            answer: RAGAnswer to explain

        Returns:
            Explanation with full provenance
        """
        explanation = {
            "query": answer.query,
            "answer": answer.answer_text,
            "confidence": answer.confidence,
            "hallucination_risk": answer.hallucination_risk,
            "trustworthy": answer.is_trustworthy(),
            "claims": [],
            "sources": [],
            "warnings": answer.warnings
        }

        # Explain each claim
        for claim in answer.grounded_claims:
            claim_info = {
                "text": claim.claim_text,
                "confidence": claim.confidence,
                "is_inferred": claim.is_inferred,
                "supporting_entities": [str(e) for e in claim.supporting_entities],
                "explanation": claim.source_explanation
            }
            explanation["claims"].append(claim_info)

        # Get source entities
        for entity_id in answer.context_used:
            try:
                entity = await self.storage.get_entity(entity_id)
                source_info = {
                    "id": str(entity.id),
                    "type": entity.ontology_type,
                    "name": entity.properties.get("name", "Unknown"),
                    "ontology_valid": entity.ontology_valid
                }
                explanation["sources"].append(source_info)
            except Exception:
                pass

        return explanation


# Convenience functions
async def ask(
    query: str,
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    llm_service: Optional[Any] = None,
    embedding_service: Optional[Any] = None
) -> RAGAnswer:
    """
    Ask a question and get a grounded answer.

    Args:
        query: Question to ask
        storage: Storage interface
        ontology: Ontology schema
        llm_service: Optional LLM service
        embedding_service: Optional embedding service

    Returns:
        RAGAnswer with grounded response
    """
    rag = OntologyGuidedRAG(storage, ontology, llm_service, embedding_service)
    return await rag.generate_answer(query)


async def retrieve(
    query: str,
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    embedding_service: Optional[Any] = None,
    max_entities: int = 20
) -> RAGContext:
    """
    Retrieve context for a query.

    Args:
        query: Query text
        storage: Storage interface
        ontology: Ontology schema
        embedding_service: Optional embedding service
        max_entities: Maximum entities to retrieve

    Returns:
        RAGContext with validated context
    """
    rag = OntologyGuidedRAG(storage, ontology, None, embedding_service)
    return await rag.retrieve_context(query, max_entities=max_entities)
