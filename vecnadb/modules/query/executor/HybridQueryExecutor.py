"""
VecnaDB Hybrid Query Executor

This module implements the query execution logic for VecnaDB's hybrid search,
combining vector similarity with graph traversal while enforcing ontology constraints.

Key Features:
- Ontology-constrained search
- Combined vector + graph ranking
- Mandatory explainability
- Bounded execution
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from vecnadb.modules.query.models.HybridQuery import (
    HybridQuery,
    SearchResult,
    SearchResultItem,
    ExecutionMetadata,
    OutputFormat,
    RankingMetric,
)
from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
    Relation,
    Subgraph,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema


class HybridQueryExecutor:
    """
    Executes hybrid queries that combine vector search and graph traversal.

    Process:
    1. Embed query text (if needed)
    2. Vector search for semantic candidates
    3. Ontology filtering
    4. Graph expansion from top candidates
    5. Combined ranking (vector + graph scores)
    6. Generate explanations
    7. Return bounded result set
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        embedding_service: Optional[Any] = None
    ):
        """
        Initialize executor.

        Args:
            storage: VecnaDB storage interface
            embedding_service: Service to embed query text
        """
        self.storage = storage
        self.embedding_service = embedding_service

    async def execute(self, query: HybridQuery) -> SearchResult:
        """
        Execute a hybrid query.

        Args:
            query: The HybridQuery to execute

        Returns:
            SearchResult with ranked entities and explanations
        """
        start_time = time.time()
        metadata = {
            "total_candidates": 0,
            "ontology_filtered": 0,
            "graph_nodes_traversed": 0,
            "vector_search_time_ms": 0.0,
            "graph_traversal_time_ms": 0.0,
            "ranking_time_ms": 0.0,
        }

        # Step 1: Embed query if needed
        if query.vector_search.enabled and not query.query_vector:
            query.query_vector = await self._embed_query(query.query_text)
            if self.embedding_service:
                metadata["query_vector_model"] = getattr(
                    self.embedding_service, "model_name", "unknown"
                )

        # Step 2: Vector search (if enabled)
        vector_candidates = []
        if query.vector_search.enabled:
            vector_start = time.time()
            vector_candidates = await self._vector_search(query)
            metadata["vector_search_time_ms"] = (time.time() - vector_start) * 1000
            metadata["total_candidates"] = len(vector_candidates)

        # Step 3: Ontology filtering
        filtered_candidates = await self._ontology_filter(
            vector_candidates,
            query.ontology_filter
        )
        metadata["ontology_filtered"] = (
            len(vector_candidates) - len(filtered_candidates)
        )

        # Step 4: Graph expansion (if enabled)
        subgraph = None
        if query.graph_traversal.enabled and filtered_candidates:
            graph_start = time.time()
            subgraph = await self._expand_subgraph(
                filtered_candidates,
                query.graph_traversal
            )
            metadata["graph_traversal_time_ms"] = (time.time() - graph_start) * 1000
            metadata["graph_nodes_traversed"] = len(subgraph.nodes) if subgraph else 0

        # Step 5: Hybrid ranking
        rank_start = time.time()
        ranked_results = await self._rank_results(
            filtered_candidates,
            subgraph,
            query
        )
        metadata["ranking_time_ms"] = (time.time() - rank_start) * 1000

        # Step 6: Generate explanations
        if query.output.include_explanations:
            ranked_results = await self._add_explanations(ranked_results, query)

        # Step 7: Truncate to max_results
        final_results = ranked_results[:query.output.max_results]
        metadata["final_results"] = len(final_results)

        # Build result
        result = SearchResult(
            query=query,
            results=final_results,
            subgraph=subgraph if query.output.format == OutputFormat.SUBGRAPH else None,
            execution_metadata=ExecutionMetadata(
                execution_time_ms=(time.time() - start_time) * 1000,
                **metadata
            )
        )

        return result

    async def _embed_query(self, query_text: str) -> List[float]:
        """
        Embed query text into vector.

        Args:
            query_text: Text to embed

        Returns:
            Embedding vector
        """
        if not self.embedding_service:
            raise ValueError("Embedding service is required for query embedding")

        # Use embedding service to create query vector
        vector = await self.embedding_service.embed(query_text)
        return vector

    async def _vector_search(
        self,
        query: HybridQuery
    ) -> List[Tuple[KnowledgeEntity, float]]:
        """
        Perform vector similarity search.

        Args:
            query: The HybridQuery

        Returns:
            List of (entity, similarity_score) tuples
        """
        if not query.query_vector:
            return []

        # Get entity types from ontology filter
        entity_types = query.ontology_filter.entity_types

        # Perform vector search
        results = await self.storage.vector_search(
            query_vector=query.query_vector,
            entity_types=entity_types,
            top_k=query.vector_search.top_k,
            similarity_threshold=query.vector_search.similarity_threshold
        )

        return results

    async def _ontology_filter(
        self,
        candidates: List[Tuple[KnowledgeEntity, float]],
        ontology_filter: Any
    ) -> List[Tuple[KnowledgeEntity, float]]:
        """
        Filter candidates by ontology constraints.

        Args:
            candidates: List of (entity, score) tuples
            ontology_filter: Ontology filter configuration

        Returns:
            Filtered list of candidates
        """
        filtered = []

        for entity, score in candidates:
            # Check ontology validity
            if ontology_filter.require_ontology_valid and not entity.ontology_valid:
                continue

            # Check entity type exclusions
            if entity.ontology_type in ontology_filter.exclude_types:
                continue

            # Check entity type inclusions (if specified)
            if ontology_filter.entity_types:
                if entity.ontology_type not in ontology_filter.entity_types:
                    continue

            filtered.append((entity, score))

        return filtered

    async def _expand_subgraph(
        self,
        candidates: List[Tuple[KnowledgeEntity, float]],
        graph_config: Any
    ) -> Optional[Subgraph]:
        """
        Expand graph from top candidates.

        Args:
            candidates: Vector search candidates
            graph_config: Graph traversal configuration

        Returns:
            Subgraph with expanded nodes and edges
        """
        if not candidates:
            return None

        # Use top candidates as seed nodes
        seed_count = min(10, len(candidates))
        seed_nodes = [entity.id for entity, _ in candidates[:seed_count]]

        # Extract subgraph
        from vecnadb.infrastructure.storage.VecnaDBStorageInterface import SubgraphFilters

        filters = SubgraphFilters(
            entity_types=None,  # No additional filtering
            relation_types=graph_config.relation_types,
            max_nodes=graph_config.max_nodes,
            max_edges=graph_config.max_edges
        )

        subgraph = await self.storage.extract_subgraph(
            seed_nodes=seed_nodes,
            max_depth=graph_config.max_depth,
            filters=filters
        )

        return subgraph

    async def _rank_results(
        self,
        candidates: List[Tuple[KnowledgeEntity, float]],
        subgraph: Optional[Subgraph],
        query: HybridQuery
    ) -> List[SearchResultItem]:
        """
        Rank results using combined vector and graph scores.

        Args:
            candidates: Vector search candidates with similarity scores
            subgraph: Expanded subgraph (if available)
            query: The HybridQuery

        Returns:
            Ranked list of SearchResultItems
        """
        ranked_items = []

        # Build entity -> similarity score mapping
        similarity_scores = {entity.id: score for entity, score in candidates}

        # Build entity -> graph score mapping (if subgraph available)
        graph_scores = {}
        if subgraph:
            graph_scores = await self._calculate_graph_scores(subgraph, query)

        # Combine scores for each entity
        for entity, similarity_score in candidates:
            graph_score = graph_scores.get(entity.id, 0.0)

            # Combined score
            combined_score = (
                query.ranking.vector_weight * similarity_score +
                query.ranking.graph_weight * graph_score
            )

            item = SearchResultItem(
                entity=entity,
                score=combined_score,
                similarity_score=similarity_score,
                graph_score=graph_score,
                path_from_query=None,  # Would be computed from subgraph
                explanation="",  # Added later
                metadata={}
            )

            ranked_items.append(item)

        # Sort by combined score (descending)
        ranked_items.sort(key=lambda x: x.score, reverse=True)

        return ranked_items

    async def _calculate_graph_scores(
        self,
        subgraph: Subgraph,
        query: HybridQuery
    ) -> Dict[UUID, float]:
        """
        Calculate graph-based scores for entities.

        Uses centrality measures like degree centrality.

        Args:
            subgraph: The subgraph
            query: The HybridQuery

        Returns:
            Dictionary of entity_id -> graph_score
        """
        scores = {}

        if not subgraph or not subgraph.nodes:
            return scores

        # Calculate degree centrality (simple measure)
        degree_counts = {}
        for edge in subgraph.edges:
            degree_counts[edge.source_id] = degree_counts.get(edge.source_id, 0) + 1
            degree_counts[edge.target_id] = degree_counts.get(edge.target_id, 0) + 1

        # Normalize to 0.0-1.0 range
        max_degree = max(degree_counts.values()) if degree_counts else 1
        for entity in subgraph.nodes:
            degree = degree_counts.get(entity.id, 0)
            scores[entity.id] = degree / max_degree if max_degree > 0 else 0.0

        return scores

    async def _add_explanations(
        self,
        results: List[SearchResultItem],
        query: HybridQuery
    ) -> List[SearchResultItem]:
        """
        Add human-readable explanations to results.

        Explanation includes:
        - Why this entity was retrieved (vector similarity? graph proximity?)
        - What path led to it
        - What ontology rules applied

        Args:
            results: List of SearchResultItems
            query: The HybridQuery

        Returns:
            Results with explanations added
        """
        for item in results:
            explanation_parts = []

            # Vector similarity explanation
            if item.similarity_score is not None and item.similarity_score > 0:
                explanation_parts.append(
                    f"Semantically similar to query (similarity: {item.similarity_score:.2f})"
                )

            # Graph proximity explanation
            if item.graph_score is not None and item.graph_score > 0:
                explanation_parts.append(
                    f"Connected in knowledge graph (centrality: {item.graph_score:.2f})"
                )

            # Ontology explanation
            explanation_parts.append(
                f"Entity type: {item.entity.ontology_type}"
            )

            # Combined score
            explanation_parts.append(
                f"Overall relevance: {item.score:.2f}"
            )

            item.explanation = " | ".join(explanation_parts)

        return results


# Convenience functions
async def simple_search(
    query_text: str,
    storage: VecnaDBStorageInterface,
    embedding_service: Any,
    entity_types: Optional[List[str]] = None,
    max_results: int = 10
) -> SearchResult:
    """
    Simplified search interface for common use case.

    Args:
        query_text: Text to search for
        storage: Storage interface
        embedding_service: Embedding service
        entity_types: Optional entity type filter
        max_results: Maximum results to return

    Returns:
        SearchResult
    """
    from vecnadb.modules.query.models.HybridQuery import HybridQueryBuilder

    # Build query
    builder = HybridQueryBuilder(query_text).with_max_results(max_results)

    if entity_types:
        builder.with_entity_types(entity_types)

    query = builder.build()

    # Execute
    executor = HybridQueryExecutor(storage, embedding_service)
    result = await executor.execute(query)

    return result


async def vector_only_search(
    query_text: str,
    storage: VecnaDBStorageInterface,
    embedding_service: Any,
    top_k: int = 10
) -> SearchResult:
    """
    Vector-only search (no graph traversal).

    Args:
        query_text: Text to search for
        storage: Storage interface
        embedding_service: Embedding service
        top_k: Number of results

    Returns:
        SearchResult
    """
    from vecnadb.modules.query.models.HybridQuery import HybridQueryBuilder

    query = (
        HybridQueryBuilder(query_text)
        .vector_only()
        .with_max_results(top_k)
        .build()
    )

    executor = HybridQueryExecutor(storage, embedding_service)
    return await executor.execute(query)


async def graph_only_search(
    start_entity_id: UUID,
    storage: VecnaDBStorageInterface,
    max_depth: int = 2,
    max_results: int = 20
) -> SearchResult:
    """
    Graph-only search (no vector similarity).

    Args:
        start_entity_id: Starting entity for traversal
        storage: Storage interface
        max_depth: Maximum traversal depth
        max_results: Maximum results

    Returns:
        SearchResult
    """
    from vecnadb.modules.query.models.HybridQuery import HybridQueryBuilder

    query = (
        HybridQueryBuilder("")  # No query text for graph-only
        .graph_only(start_entity_id)
        .with_graph_depth(max_depth)
        .with_max_results(max_results)
        .build()
    )

    executor = HybridQueryExecutor(storage, None)  # No embedding service needed
    return await executor.execute(query)
