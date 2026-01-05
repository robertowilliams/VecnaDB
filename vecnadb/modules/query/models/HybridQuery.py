"""
VecnaDB Hybrid Query Models

This module defines the query models for VecnaDB's hybrid search system
that combines vector similarity with graph structure traversal.

Key Principles:
- Ontology-constrained search
- Mandatory explainability
- Combined vector + graph ranking
- Bounded result sets
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import (
    KnowledgeEntity,
    EmbeddingType,
)
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    Relation,
    Direction,
)


# Enums
class OutputFormat(str, Enum):
    """Output format for search results"""
    ENTITIES = "entities"      # List of entities only
    SUBGRAPH = "subgraph"      # Entities + relations as subgraph
    TEXT = "text"              # Text context from entities
    JSON = "json"              # Structured JSON


class RankingMetric(str, Enum):
    """Metrics used for ranking results"""
    SEMANTIC_SIMILARITY = "semantic_similarity"  # Vector similarity score
    CENTRALITY = "centrality"                    # Graph centrality (PageRank, etc.)
    RECENCY = "recency"                          # Based on created_at/updated_at
    RELEVANCE = "relevance"                      # Combined score
    CUSTOM = "custom"                            # User-defined scoring


# Configuration Models
class VectorSearchConfig(BaseModel):
    """Configuration for vector similarity search"""
    enabled: bool = True
    top_k: int = 50
    similarity_threshold: float = 0.7
    embedding_types: List[EmbeddingType] = Field(
        default_factory=lambda: [EmbeddingType.CONTENT]
    )
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product


class GraphTraversalConfig(BaseModel):
    """Configuration for graph traversal"""
    enabled: bool = True
    max_depth: int = 3
    relation_types: Optional[List[str]] = None  # None = all types
    direction: Direction = Direction.BOTH
    max_nodes: int = 100
    max_edges: int = 200


class OntologyFilter(BaseModel):
    """Ontology-based filtering for search"""
    ontology_id: Optional[UUID] = None  # If None, use default ontology
    entity_types: Optional[List[str]] = None  # None = all types
    exclude_types: List[str] = Field(default_factory=list)
    relation_types: Optional[List[str]] = None  # For graph traversal
    require_ontology_valid: bool = True  # Only return validated entities


class RankingConfig(BaseModel):
    """Configuration for result ranking"""
    vector_weight: float = 0.7  # Weight for vector similarity
    graph_weight: float = 0.3   # Weight for graph proximity
    rank_by: List[RankingMetric] = Field(
        default_factory=lambda: [
            RankingMetric.SEMANTIC_SIMILARITY,
            RankingMetric.CENTRALITY
        ]
    )
    custom_scoring_fn: Optional[str] = None  # Name of custom function


class OutputConfig(BaseModel):
    """Configuration for output format"""
    format: OutputFormat = OutputFormat.SUBGRAPH
    max_results: int = 20
    include_paths: bool = True  # Include graph paths in results
    include_explanations: bool = True  # Mandatory explainability
    include_scores: bool = True  # Include similarity/relevance scores
    include_metadata: bool = True  # Include entity metadata


# Main Query Model
class HybridQuery(BaseModel):
    """
    Hybrid query combining vector search and graph traversal.

    This is the primary query interface for VecnaDB, enabling:
    - Semantic search via vector similarity
    - Structural search via graph traversal
    - Ontology-constrained filtering
    - Combined ranking
    - Explainable results
    """

    # Query input
    query_text: str
    query_vector: Optional[List[float]] = None  # Pre-computed or will be embedded

    # Search configurations
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    graph_traversal: GraphTraversalConfig = Field(default_factory=GraphTraversalConfig)

    # Filtering
    ontology_filter: OntologyFilter = Field(default_factory=OntologyFilter)

    # Ranking
    ranking: RankingConfig = Field(default_factory=RankingConfig)

    # Output
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Optional metadata
    query_id: Optional[UUID] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def validate_config(self) -> List[str]:
        """
        Validate query configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check that at least one search type is enabled
        if not self.vector_search.enabled and not self.graph_traversal.enabled:
            errors.append("At least one of vector_search or graph_traversal must be enabled")

        # Check weights sum to reasonable value
        total_weight = self.ranking.vector_weight + self.ranking.graph_weight
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Ranking weights should sum to 1.0, got {total_weight}")

        # Check top_k is reasonable
        if self.vector_search.top_k > 1000:
            errors.append("vector_search.top_k should not exceed 1000")

        # Check max_results is reasonable
        if self.output.max_results > 500:
            errors.append("output.max_results should not exceed 500")

        return errors


# Result Models
class SearchResultItem(BaseModel):
    """Individual search result with scores and explanation"""
    entity: KnowledgeEntity
    score: float  # Combined relevance score
    similarity_score: Optional[float] = None  # Vector similarity (0.0-1.0)
    graph_score: Optional[float] = None  # Graph proximity score (0.0-1.0)
    path_from_query: Optional[List[Relation]] = None  # Graph path to this entity
    explanation: str = ""  # Human-readable explanation of why this was returned
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionMetadata(BaseModel):
    """Metadata about query execution"""
    execution_time_ms: float
    vector_search_time_ms: float = 0.0
    graph_traversal_time_ms: float = 0.0
    ranking_time_ms: float = 0.0
    total_candidates: int = 0
    ontology_filtered: int = 0
    final_results: int = 0
    query_vector_model: Optional[str] = None
    graph_nodes_traversed: int = 0


class SearchResult(BaseModel):
    """
    Complete search result with entities, scores, and explanations.

    Every search result is explainable - users can understand why
    each entity was returned.
    """
    query: HybridQuery
    results: List[SearchResultItem]
    subgraph: Optional[Any] = None  # Subgraph if output.format == SUBGRAPH
    execution_metadata: ExecutionMetadata

    def get_top_entities(self, n: int = 10) -> List[KnowledgeEntity]:
        """Get top N entities by score"""
        return [item.entity for item in self.results[:n]]

    def get_entities_by_type(self, entity_type: str) -> List[KnowledgeEntity]:
        """Filter results by entity type"""
        return [
            item.entity
            for item in self.results
            if item.entity.ontology_type == entity_type
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()


# Query Builder (Convenience)
class HybridQueryBuilder:
    """
    Fluent builder for creating HybridQuery instances.

    Example:
        query = (HybridQueryBuilder("What is machine learning?")
                .with_entity_types(["Concept", "Document"])
                .with_max_results(10)
                .with_graph_depth(2)
                .build())
    """

    def __init__(self, query_text: str):
        self.query_text = query_text
        self._vector_config = VectorSearchConfig()
        self._graph_config = GraphTraversalConfig()
        self._ontology_filter = OntologyFilter()
        self._ranking_config = RankingConfig()
        self._output_config = OutputConfig()
        self.query_vector = None
        self.metadata = {}

    def with_vector(self, vector: List[float]) -> "HybridQueryBuilder":
        """Set pre-computed query vector"""
        self.query_vector = vector
        return self

    def with_entity_types(self, entity_types: List[str]) -> "HybridQueryBuilder":
        """Filter by entity types"""
        self._ontology_filter.entity_types = entity_types
        return self

    def exclude_entity_types(self, entity_types: List[str]) -> "HybridQueryBuilder":
        """Exclude entity types"""
        self._ontology_filter.exclude_types = entity_types
        return self

    def with_relation_types(self, relation_types: List[str]) -> "HybridQueryBuilder":
        """Filter by relation types for graph traversal"""
        self._graph_config.relation_types = relation_types
        return self

    def with_top_k(self, top_k: int) -> "HybridQueryBuilder":
        """Set number of vector search candidates"""
        self._vector_config.top_k = top_k
        return self

    def with_max_results(self, max_results: int) -> "HybridQueryBuilder":
        """Set maximum number of final results"""
        self._output_config.max_results = max_results
        return self

    def with_graph_depth(self, max_depth: int) -> "HybridQueryBuilder":
        """Set maximum graph traversal depth"""
        self._graph_config.max_depth = max_depth
        return self

    def with_similarity_threshold(self, threshold: float) -> "HybridQueryBuilder":
        """Set minimum similarity threshold"""
        self._vector_config.similarity_threshold = threshold
        return self

    def with_ranking_weights(
        self,
        vector_weight: float,
        graph_weight: float
    ) -> "HybridQueryBuilder":
        """Set ranking weights for vector vs graph"""
        self._ranking_config.vector_weight = vector_weight
        self._ranking_config.graph_weight = graph_weight
        return self

    def with_output_format(self, format: OutputFormat) -> "HybridQueryBuilder":
        """Set output format"""
        self._output_config.format = format
        return self

    def vector_only(self) -> "HybridQueryBuilder":
        """Disable graph traversal, use vector search only"""
        self._graph_config.enabled = False
        self._ranking_config.vector_weight = 1.0
        self._ranking_config.graph_weight = 0.0
        return self

    def graph_only(self, start_entity_id: UUID) -> "HybridQueryBuilder":
        """Disable vector search, use graph traversal only"""
        self._vector_config.enabled = False
        self._ranking_config.vector_weight = 0.0
        self._ranking_config.graph_weight = 1.0
        self.metadata["start_entity_id"] = str(start_entity_id)
        return self

    def build(self) -> HybridQuery:
        """Build the HybridQuery"""
        query = HybridQuery(
            query_text=self.query_text,
            query_vector=self.query_vector,
            vector_search=self._vector_config,
            graph_traversal=self._graph_config,
            ontology_filter=self._ontology_filter,
            ranking=self._ranking_config,
            output=self._output_config,
            metadata=self.metadata
        )

        # Validate
        errors = query.validate_config()
        if errors:
            raise ValueError(f"Invalid query configuration: {', '.join(errors)}")

        return query
