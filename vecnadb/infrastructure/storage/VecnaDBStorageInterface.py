"""
VecnaDB Unified Storage Interface

This module defines the unified storage interface for VecnaDB that enforces:
- Dual representation (graph + vector)
- Ontology validation
- Atomic operations
- Full auditability

This interface replaces the separate GraphDBInterface and VectorDBInterface
with a single, coherent API that maintains VecnaDB's core principles.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from enum import Enum

from vecnadb.infrastructure.engine.models.KnowledgeEntity import (
    KnowledgeEntity,
    EmbeddingRecord,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema
from vecnadb.modules.ontology.validation.OntologyValidator import ValidationResult


class Direction(str, Enum):
    """Direction for relation queries"""
    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class Relation:
    """Represents a relation between two entities"""
    def __init__(
        self,
        id: UUID,
        source_id: UUID,
        relation_type: str,
        target_id: UUID,
        properties: Optional[Dict[str, Any]] = None,
        inferred: bool = False
    ):
        self.id = id
        self.source_id = source_id
        self.relation_type = relation_type
        self.target_id = target_id
        self.properties = properties or {}
        self.inferred = inferred


class Subgraph:
    """Represents a bounded subgraph"""
    def __init__(
        self,
        nodes: List[KnowledgeEntity],
        edges: List[Relation],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.nodes = nodes
        self.edges = edges
        self.metadata = metadata or {}


class SubgraphFilters:
    """Filters for subgraph extraction"""
    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        max_nodes: int = 100,
        max_edges: int = 200
    ):
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.max_nodes = max_nodes
        self.max_edges = max_edges


class VecnaDBStorageInterface(ABC):
    """
    Unified storage interface for VecnaDB.

    This interface abstracts over different storage backends while
    enforcing VecnaDB's core principles:
    1. Dual representation (every entity has graph node + vector embedding)
    2. Ontology validation (all entities conform to schema)
    3. Atomic operations (graph + vector updates are atomic)
    4. Full auditability (all operations are logged)

    Implementations:
    - LanceDBKuzuAdapter: LanceDB (vector) + Kuzu (graph)
    - Neo4jAdapter: Neo4j with native vector support
    - NeptuneAnalyticsAdapter: AWS Neptune Analytics (native hybrid)
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    @abstractmethod
    async def initialize(self):
        """Initialize the storage backend"""
        pass

    @abstractmethod
    async def close(self):
        """Close connections and cleanup resources"""
        pass

    # ========================================================================
    # ONTOLOGY MANAGEMENT
    # ========================================================================

    @abstractmethod
    async def register_ontology(self, ontology: OntologySchema) -> UUID:
        """
        Register a new ontology schema.

        Args:
            ontology: The ontology schema to register

        Returns:
            UUID of the registered ontology

        Raises:
            ValueError: If ontology with same name+version already exists
        """
        pass

    @abstractmethod
    async def get_ontology(self, ontology_id: UUID) -> Optional[OntologySchema]:
        """
        Get an ontology schema by ID.

        Args:
            ontology_id: UUID of the ontology

        Returns:
            OntologySchema if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_ontology_by_name_version(
        self,
        name: str,
        version: str
    ) -> Optional[OntologySchema]:
        """Get ontology by name and version"""
        pass

    @abstractmethod
    async def list_ontologies(self) -> List[OntologySchema]:
        """List all registered ontologies"""
        pass

    @abstractmethod
    async def get_default_ontology(self) -> OntologySchema:
        """Get the default (core) ontology"""
        pass

    # ========================================================================
    # ENTITY OPERATIONS
    # ========================================================================

    @abstractmethod
    async def add_entity(
        self,
        entity: KnowledgeEntity,
        validate: bool = True
    ) -> UUID:
        """
        Add a new knowledge entity.

        This operation is atomic - both graph and vector representations
        are created together or the operation fails.

        Process:
        1. Validate entity against ontology (if validate=True)
        2. Add node to graph database
        3. Index all embeddings in vector database
        4. Update entity.ontology_valid flag
        5. Return entity ID

        Args:
            entity: The KnowledgeEntity to add
            validate: Whether to validate against ontology (default: True)

        Returns:
            UUID of the added entity

        Raises:
            ValidationError: If entity fails ontology validation
            DualRepresentationError: If entity lacks graph_node_id or embeddings
            StorageError: If storage operation fails
        """
        pass

    @abstractmethod
    async def add_entities(
        self,
        entities: List[KnowledgeEntity],
        validate: bool = True
    ) -> List[UUID]:
        """
        Batch add multiple entities.

        All entities are validated and added atomically.
        If any entity fails validation, entire batch is rejected.

        Args:
            entities: List of KnowledgeEntities to add
            validate: Whether to validate against ontology

        Returns:
            List of UUIDs of added entities
        """
        pass

    @abstractmethod
    async def get_entity(self, entity_id: UUID) -> Optional[KnowledgeEntity]:
        """
        Retrieve an entity by ID.

        Returns complete entity with all embeddings.

        Args:
            entity_id: UUID of the entity

        Returns:
            KnowledgeEntity if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_entities(self, entity_ids: List[UUID]) -> List[KnowledgeEntity]:
        """Batch retrieve multiple entities"""
        pass

    @abstractmethod
    async def update_entity(
        self,
        entity: KnowledgeEntity,
        validate: bool = True
    ) -> None:
        """
        Update an existing entity.

        Creates a new version of the entity (preserves history).

        Args:
            entity: The updated KnowledgeEntity
            validate: Whether to validate against ontology

        Raises:
            NotFoundError: If entity doesn't exist
            ValidationError: If updated entity fails validation
        """
        pass

    @abstractmethod
    async def delete_entity(
        self,
        entity_id: UUID,
        soft_delete: bool = True
    ) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: UUID of entity to delete
            soft_delete: If True, mark as deleted but preserve data

        Returns:
            True if entity was deleted, False if not found
        """
        pass

    # ========================================================================
    # RELATION OPERATIONS
    # ========================================================================

    @abstractmethod
    async def add_relation(
        self,
        source_id: UUID,
        relation_type: str,
        target_id: UUID,
        properties: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> UUID:
        """
        Add a relation between two entities.

        Process:
        1. Validate source and target entities exist
        2. Validate relation against ontology (if validate=True)
        3. Check cardinality constraints
        4. Add edge to graph database
        5. Optionally create triplet embedding
        6. Return relation ID

        Args:
            source_id: UUID of source entity
            relation_type: Type of relation (must exist in ontology)
            target_id: UUID of target entity
            properties: Optional properties on the relation
            validate: Whether to validate against ontology

        Returns:
            UUID of the created relation

        Raises:
            NotFoundError: If source or target entity doesn't exist
            ValidationError: If relation fails ontology validation
            CardinalityError: If relation violates cardinality constraints
        """
        pass

    @abstractmethod
    async def add_relations(
        self,
        relations: List[Tuple[UUID, str, UUID, Optional[Dict]]],
        validate: bool = True
    ) -> List[UUID]:
        """
        Batch add multiple relations.

        Args:
            relations: List of (source_id, relation_type, target_id, properties) tuples
            validate: Whether to validate against ontology

        Returns:
            List of UUIDs of created relations
        """
        pass

    @abstractmethod
    async def get_relations(
        self,
        entity_id: UUID,
        relation_type: Optional[str] = None,
        direction: Direction = Direction.BOTH
    ) -> List[Relation]:
        """
        Get relations for an entity.

        Args:
            entity_id: UUID of the entity
            relation_type: Optional filter by relation type
            direction: OUTGOING, INCOMING, or BOTH

        Returns:
            List of relations
        """
        pass

    @abstractmethod
    async def delete_relation(self, relation_id: UUID) -> bool:
        """
        Delete a relation.

        Args:
            relation_id: UUID of relation to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    # ========================================================================
    # EMBEDDING OPERATIONS
    # ========================================================================

    @abstractmethod
    async def update_embeddings(
        self,
        entity_id: UUID,
        embeddings: List[EmbeddingRecord]
    ) -> None:
        """
        Update embeddings for an entity.

        Supports re-embedding with new models without data loss.

        Args:
            entity_id: UUID of the entity
            embeddings: New embeddings to add/replace

        Raises:
            NotFoundError: If entity doesn't exist
        """
        pass

    @abstractmethod
    async def get_embeddings(
        self,
        entity_id: UUID,
        embedding_type: Optional[str] = None
    ) -> List[EmbeddingRecord]:
        """
        Get embeddings for an entity.

        Args:
            entity_id: UUID of the entity
            embedding_type: Optional filter by embedding type

        Returns:
            List of embedding records
        """
        pass

    # ========================================================================
    # HYBRID SEARCH
    # ========================================================================

    @abstractmethod
    async def vector_search(
        self,
        query_vector: List[float],
        entity_types: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[KnowledgeEntity, float]]:
        """
        Perform vector similarity search.

        Args:
            query_vector: Query embedding vector
            entity_types: Optional filter by entity types (from ontology)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of (entity, similarity_score) tuples, sorted by score descending
        """
        pass

    @abstractmethod
    async def graph_search(
        self,
        start_entity_id: UUID,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 3,
        entity_type_filter: Optional[List[str]] = None
    ) -> Subgraph:
        """
        Perform graph traversal search.

        Args:
            start_entity_id: Starting entity for traversal
            relation_types: Optional filter by relation types
            max_depth: Maximum traversal depth
            entity_type_filter: Optional filter by entity types

        Returns:
            Subgraph containing reachable entities and relations
        """
        pass

    @abstractmethod
    async def extract_subgraph(
        self,
        seed_nodes: List[UUID],
        max_depth: int = 2,
        filters: Optional[SubgraphFilters] = None
    ) -> Subgraph:
        """
        Extract a bounded subgraph around seed nodes.

        Used for context extraction in RAG.

        Args:
            seed_nodes: Starting entities
            max_depth: Maximum graph distance from seeds
            filters: Optional filters for nodes and edges

        Returns:
            Bounded subgraph
        """
        pass

    # ========================================================================
    # VERSIONING
    # ========================================================================

    @abstractmethod
    async def get_entity_history(
        self,
        entity_id: UUID
    ) -> List[KnowledgeEntity]:
        """
        Get all versions of an entity.

        Returns versions in chronological order (oldest to newest).

        Args:
            entity_id: UUID of the entity

        Returns:
            List of entity versions
        """
        pass

    @abstractmethod
    async def get_entity_at_version(
        self,
        entity_id: UUID,
        version: int
    ) -> Optional[KnowledgeEntity]:
        """
        Get a specific version of an entity.

        Args:
            entity_id: UUID of the entity
            version: Version number

        Returns:
            Entity at that version, or None if not found
        """
        pass

    # ========================================================================
    # STATISTICS AND INTROSPECTION
    # ========================================================================

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with stats like:
            - total_entities
            - total_relations
            - total_embeddings
            - entities_by_type
            - relations_by_type
        """
        pass

    @abstractmethod
    async def get_entity_count(
        self,
        entity_type: Optional[str] = None
    ) -> int:
        """
        Count entities, optionally filtered by type.

        Args:
            entity_type: Optional entity type filter

        Returns:
            Number of entities
        """
        pass

    @abstractmethod
    async def get_relation_count(
        self,
        relation_type: Optional[str] = None
    ) -> int:
        """
        Count relations, optionally filtered by type.

        Args:
            relation_type: Optional relation type filter

        Returns:
            Number of relations
        """
        pass


# Custom Exceptions
class ValidationError(Exception):
    """Raised when entity/relation fails ontology validation"""
    pass


class DualRepresentationError(Exception):
    """Raised when entity violates dual representation requirement"""
    pass


class CardinalityError(Exception):
    """Raised when relation violates cardinality constraints"""
    pass


class NotFoundError(Exception):
    """Raised when entity/relation not found"""
    pass


class StorageError(Exception):
    """Generic storage operation error"""
    pass
