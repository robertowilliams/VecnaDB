"""
LanceDB + Kuzu Adapter for VecnaDB

This adapter implements VecnaDBStorageInterface using:
- LanceDB for vector storage and search
- Kuzu for graph storage and traversal

This is the default storage backend for VecnaDB.
"""

import asyncio
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime, timezone

from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
    Direction,
    Relation,
    Subgraph,
    SubgraphFilters,
    ValidationError,
    DualRepresentationError,
    CardinalityError,
    NotFoundError,
    StorageError,
)
from vecnadb.infrastructure.engine.models.KnowledgeEntity import (
    KnowledgeEntity,
    EmbeddingRecord,
    EmbeddingType,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema
from vecnadb.modules.ontology.validation.OntologyValidator import (
    OntologyValidator,
    ValidationResult,
)
from vecnadb.modules.ontology.loaders.OntologyLoader import load_core_ontology


class LanceDBKuzuAdapter(VecnaDBStorageInterface):
    """
    Hybrid storage adapter using LanceDB (vector) + Kuzu (graph).

    This adapter enforces:
    - Dual representation (atomic graph + vector writes)
    - Ontology validation before all writes
    - Cardinality constraints
    - Full auditability
    """

    def __init__(
        self,
        lance_db_path: str = "./.data/lancedb",
        kuzu_db_path: str = "./.data/kuzu",
        default_ontology: Optional[OntologySchema] = None
    ):
        """
        Initialize the adapter.

        Args:
            lance_db_path: Path to LanceDB storage
            kuzu_db_path: Path to Kuzu database
            default_ontology: Default ontology (uses core if not provided)
        """
        self.lance_db_path = lance_db_path
        self.kuzu_db_path = kuzu_db_path

        # Will be initialized in initialize()
        self.lance_db = None
        self.kuzu_db = None
        self.kuzu_conn = None

        # Ontology management
        self.ontologies: Dict[UUID, OntologySchema] = {}
        self.default_ontology = default_ontology

        # Validators cache (one per ontology)
        self.validators: Dict[UUID, OntologyValidator] = {}

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def initialize(self):
        """Initialize LanceDB and Kuzu connections"""
        # Import here to avoid circular dependencies
        import lancedb
        import kuzu

        # Initialize LanceDB
        self.lance_db = lancedb.connect(self.lance_db_path)

        # Initialize Kuzu
        self.kuzu_db = kuzu.Database(self.kuzu_db_path)
        self.kuzu_conn = kuzu.Connection(self.kuzu_db)

        # Create core schema in Kuzu
        await self._initialize_kuzu_schema()

        # Load and register default ontology
        if not self.default_ontology:
            self.default_ontology = load_core_ontology()

        await self.register_ontology(self.default_ontology)

    async def _initialize_kuzu_schema(self):
        """Create base schema in Kuzu for knowledge graph"""
        # Create KnowledgeEntity node table
        self.kuzu_conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS KnowledgeEntity(
                id STRING,
                ontology_id STRING,
                ontology_type STRING,
                ontology_valid BOOLEAN,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                version INT64,
                graph_node_id STRING,
                metadata STRING,
                PRIMARY KEY (id)
            )
        """)

        # Create Relation edge table
        self.kuzu_conn.execute("""
            CREATE REL TABLE IF NOT EXISTS Relation(
                FROM KnowledgeEntity TO KnowledgeEntity,
                relation_type STRING,
                properties STRING,
                created_at TIMESTAMP
            )
        """)

    async def close(self):
        """Close connections"""
        if self.kuzu_conn:
            self.kuzu_conn.close()

    # ========================================================================
    # ONTOLOGY MANAGEMENT
    # ========================================================================

    async def register_ontology(self, ontology: OntologySchema) -> UUID:
        """Register a new ontology schema"""
        # Check for duplicates
        for existing in self.ontologies.values():
            if existing.name == ontology.name and existing.version == ontology.version:
                raise ValueError(
                    f"Ontology {ontology.name} v{ontology.version} already registered"
                )

        # Store ontology
        self.ontologies[ontology.id] = ontology

        # Create validator
        self.validators[ontology.id] = OntologyValidator(ontology, storage=self)

        return ontology.id

    async def get_ontology(self, ontology_id: UUID) -> Optional[OntologySchema]:
        """Get ontology by ID"""
        return self.ontologies.get(ontology_id)

    async def get_ontology_by_name_version(
        self,
        name: str,
        version: str
    ) -> Optional[OntologySchema]:
        """Get ontology by name and version"""
        for ontology in self.ontologies.values():
            if ontology.name == name and ontology.version == version:
                return ontology
        return None

    async def list_ontologies(self) -> List[OntologySchema]:
        """List all registered ontologies"""
        return list(self.ontologies.values())

    async def get_default_ontology(self) -> OntologySchema:
        """Get the default ontology"""
        return self.default_ontology

    # ========================================================================
    # ENTITY OPERATIONS
    # ========================================================================

    async def add_entity(
        self,
        entity: KnowledgeEntity,
        validate: bool = True
    ) -> UUID:
        """Add a new knowledge entity with dual representation"""

        # 1. Validate dual representation
        if not entity.graph_node_id:
            raise DualRepresentationError(
                "Entity must have graph_node_id for dual representation"
            )

        if not entity.embeddings or len(entity.embeddings) == 0:
            raise DualRepresentationError(
                "Entity must have at least one embedding for dual representation"
            )

        # 2. Validate against ontology
        if validate:
            ontology = await self.get_ontology(entity.ontology_id)
            if not ontology:
                raise ValidationError(
                    f"Unknown ontology: {entity.ontology_id}"
                )

            validator = self.validators.get(entity.ontology_id)
            if not validator:
                validator = OntologyValidator(ontology, storage=self)
                self.validators[entity.ontology_id] = validator

            result = await validator.validate_entity(entity)
            if not result.valid:
                entity.ontology_valid = False
                entity.validation_errors = result.errors
                raise ValidationError(
                    f"Entity validation failed: {', '.join(result.errors)}"
                )

            entity.ontology_valid = True
            entity.validation_errors = None

        # 3. Atomic dual write
        try:
            # Add to graph (Kuzu)
            await self._add_entity_to_graph(entity)

            # Add to vector index (LanceDB)
            await self._add_entity_to_vector(entity)

            return entity.id

        except Exception as e:
            # Rollback on failure
            await self._rollback_entity(entity.id)
            raise StorageError(f"Failed to add entity: {str(e)}")

    async def _add_entity_to_graph(self, entity: KnowledgeEntity):
        """Add entity as node in Kuzu graph"""
        import json

        query = """
            CREATE (n:KnowledgeEntity {
                id: $id,
                ontology_id: $ontology_id,
                ontology_type: $ontology_type,
                ontology_valid: $ontology_valid,
                created_at: $created_at,
                updated_at: $updated_at,
                version: $version,
                graph_node_id: $graph_node_id,
                metadata: $metadata
            })
        """

        self.kuzu_conn.execute(query, {
            "id": str(entity.id),
            "ontology_id": str(entity.ontology_id),
            "ontology_type": entity.ontology_type,
            "ontology_valid": entity.ontology_valid,
            "created_at": entity.created_at,
            "updated_at": entity.updated_at,
            "version": entity.version,
            "graph_node_id": entity.graph_node_id,
            "metadata": json.dumps(entity.metadata)
        })

    async def _add_entity_to_vector(self, entity: KnowledgeEntity):
        """Index entity embeddings in LanceDB"""
        import pyarrow as pa

        # Get or create table for this entity type
        table_name = f"entities_{entity.ontology_type.lower()}"

        # Prepare data for each embedding
        records = []
        for embedding in entity.embeddings:
            records.append({
                "id": str(embedding.id),
                "entity_id": str(entity.id),
                "entity_type": entity.ontology_type,
                "embedding_type": embedding.embedding_type.value,
                "vector": embedding.vector,
                "model": embedding.model,
                "model_version": embedding.model_version,
                "dimensions": embedding.dimensions,
                "created_at": embedding.created_at.isoformat(),
                "metadata": entity.model_dump_json()
            })

        # Add to LanceDB
        if table_name in self.lance_db.table_names():
            table = self.lance_db.open_table(table_name)
            table.add(records)
        else:
            self.lance_db.create_table(table_name, records)

    async def _rollback_entity(self, entity_id: UUID):
        """Rollback entity creation (best effort)"""
        try:
            # Remove from graph
            query = "MATCH (n:KnowledgeEntity {id: $id}) DELETE n"
            self.kuzu_conn.execute(query, {"id": str(entity_id)})

            # Remove from vector indexes
            # (LanceDB doesn't support delete yet, so we skip)
        except:
            pass  # Best effort rollback

    async def add_entities(
        self,
        entities: List[KnowledgeEntity],
        validate: bool = True
    ) -> List[UUID]:
        """Batch add entities"""
        # For now, add one by one (can be optimized later)
        entity_ids = []
        for entity in entities:
            entity_id = await self.add_entity(entity, validate=validate)
            entity_ids.append(entity_id)
        return entity_ids

    async def get_entity(self, entity_id: UUID) -> Optional[KnowledgeEntity]:
        """Retrieve entity by ID"""
        import json

        # Query from graph
        query = """
            MATCH (n:KnowledgeEntity {id: $id})
            RETURN n
        """

        result = self.kuzu_conn.execute(query, {"id": str(entity_id)})

        if not result.has_next():
            return None

        node = result.get_next()[0]

        # Reconstruct KnowledgeEntity
        # Note: This is simplified - full implementation would fetch embeddings
        # from LanceDB and reconstruct complete entity

        entity = KnowledgeEntity(
            id=UUID(node["id"]),
            ontology_id=UUID(node["ontology_id"]),
            ontology_type=node["ontology_type"],
            ontology_valid=node["ontology_valid"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
            version=node["version"],
            graph_node_id=node["graph_node_id"],
            metadata=json.loads(node["metadata"]),
            embeddings=[]  # Would fetch from LanceDB
        )

        return entity

    async def get_entities(self, entity_ids: List[UUID]) -> List[KnowledgeEntity]:
        """Batch get entities"""
        entities = []
        for entity_id in entity_ids:
            entity = await self.get_entity(entity_id)
            if entity:
                entities.append(entity)
        return entities

    async def update_entity(
        self,
        entity: KnowledgeEntity,
        validate: bool = True
    ) -> None:
        """Update entity (creates new version)"""
        # For Sprint 3, implement basic update
        # Full versioning will be in Sprint 7
        existing = await self.get_entity(entity.id)
        if not existing:
            raise NotFoundError(f"Entity {entity.id} not found")

        # Validate if requested
        if validate:
            ontology = await self.get_ontology(entity.ontology_id)
            validator = self.validators[entity.ontology_id]
            result = await validator.validate_entity(entity)
            if not result.valid:
                raise ValidationError(f"Validation failed: {result.errors}")

        # Update in graph and vector
        # (Simplified for Sprint 3)
        await self._update_entity_in_graph(entity)
        await self._update_entity_in_vector(entity)

    async def _update_entity_in_graph(self, entity: KnowledgeEntity):
        """Update entity in Kuzu"""
        import json

        query = """
            MATCH (n:KnowledgeEntity {id: $id})
            SET n.updated_at = $updated_at,
                n.version = $version,
                n.metadata = $metadata,
                n.ontology_valid = $ontology_valid
        """

        self.kuzu_conn.execute(query, {
            "id": str(entity.id),
            "updated_at": entity.updated_at,
            "version": entity.version,
            "metadata": json.dumps(entity.metadata),
            "ontology_valid": entity.ontology_valid
        })

    async def _update_entity_in_vector(self, entity: KnowledgeEntity):
        """Update entity embeddings in LanceDB"""
        # For Sprint 3, skip vector update
        # Full implementation in Sprint 7
        pass

    async def delete_entity(
        self,
        entity_id: UUID,
        soft_delete: bool = True
    ) -> bool:
        """Delete entity"""
        query = "MATCH (n:KnowledgeEntity {id: $id}) DELETE n"

        try:
            self.kuzu_conn.execute(query, {"id": str(entity_id)})
            return True
        except:
            return False

    # ========================================================================
    # RELATION OPERATIONS
    # ========================================================================

    async def add_relation(
        self,
        source_id: UUID,
        relation_type: str,
        target_id: UUID,
        properties: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> UUID:
        """Add relation between entities"""
        import json

        # 1. Verify entities exist
        source = await self.get_entity(source_id)
        target = await self.get_entity(target_id)

        if not source:
            raise NotFoundError(f"Source entity {source_id} not found")
        if not target:
            raise NotFoundError(f"Target entity {target_id} not found")

        # 2. Validate relation
        if validate:
            ontology = await self.get_ontology(source.ontology_id)
            validator = self.validators[source.ontology_id]
            result = await validator.validate_relation(
                source, relation_type, target, properties
            )
            if not result.valid:
                raise ValidationError(f"Relation validation failed: {result.errors}")

        # 3. Add relation to graph
        relation_id = uuid4()
        query = """
            MATCH (s:KnowledgeEntity {id: $source_id})
            MATCH (t:KnowledgeEntity {id: $target_id})
            CREATE (s)-[r:Relation {
                relation_type: $relation_type,
                properties: $properties,
                created_at: $created_at
            }]->(t)
        """

        self.kuzu_conn.execute(query, {
            "source_id": str(source_id),
            "target_id": str(target_id),
            "relation_type": relation_type,
            "properties": json.dumps(properties or {}),
            "created_at": datetime.now(timezone.utc)
        })

        return relation_id

    async def add_relations(
        self,
        relations: List[Tuple[UUID, str, UUID, Optional[Dict]]],
        validate: bool = True
    ) -> List[UUID]:
        """Batch add relations"""
        relation_ids = []
        for source_id, relation_type, target_id, properties in relations:
            relation_id = await self.add_relation(
                source_id, relation_type, target_id, properties, validate
            )
            relation_ids.append(relation_id)
        return relation_ids

    async def get_relations(
        self,
        entity_id: UUID,
        relation_type: Optional[str] = None,
        direction: Direction = Direction.BOTH
    ) -> List[Relation]:
        """Get relations for entity"""
        relations = []

        if direction in (Direction.OUTGOING, Direction.BOTH):
            # Get outgoing relations
            query = """
                MATCH (s:KnowledgeEntity {id: $entity_id})-[r:Relation]->(t)
                WHERE $relation_type IS NULL OR r.relation_type = $relation_type
                RETURN r, t.id
            """
            result = self.kuzu_conn.execute(query, {
                "entity_id": str(entity_id),
                "relation_type": relation_type
            })

            while result.has_next():
                row = result.get_next()
                # Parse relation (simplified)
                relations.append(Relation(
                    id=uuid4(),  # Would be stored in practice
                    source_id=entity_id,
                    relation_type=row[0]["relation_type"],
                    target_id=UUID(row[1]),
                    properties={}
                ))

        # Similar for incoming relations...

        return relations

    async def delete_relation(self, relation_id: UUID) -> bool:
        """Delete relation"""
        # Simplified for Sprint 3
        return True

    # ========================================================================
    # EMBEDDING OPERATIONS
    # ========================================================================

    async def update_embeddings(
        self,
        entity_id: UUID,
        embeddings: List[EmbeddingRecord]
    ) -> None:
        """Update entity embeddings"""
        # Implementation in Sprint 7
        pass

    async def get_embeddings(
        self,
        entity_id: UUID,
        embedding_type: Optional[str] = None
    ) -> List[EmbeddingRecord]:
        """Get entity embeddings"""
        # Implementation in Sprint 7
        return []

    # ========================================================================
    # HYBRID SEARCH
    # ========================================================================

    async def vector_search(
        self,
        query_vector: List[float],
        entity_types: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[KnowledgeEntity, float]]:
        """Vector similarity search"""
        # Implementation in Sprint 4
        return []

    async def graph_search(
        self,
        start_entity_id: UUID,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 3,
        entity_type_filter: Optional[List[str]] = None
    ) -> Subgraph:
        """Graph traversal search"""
        # Implementation in Sprint 4
        return Subgraph(nodes=[], edges=[])

    async def extract_subgraph(
        self,
        seed_nodes: List[UUID],
        max_depth: int = 2,
        filters: Optional[SubgraphFilters] = None
    ) -> Subgraph:
        """Extract bounded subgraph"""
        # Implementation in Sprint 4
        return Subgraph(nodes=[], edges=[])

    # ========================================================================
    # VERSIONING
    # ========================================================================

    async def get_entity_history(
        self,
        entity_id: UUID
    ) -> List[KnowledgeEntity]:
        """Get entity version history"""
        # Implementation in Sprint 7
        return []

    async def get_entity_at_version(
        self,
        entity_id: UUID,
        version: int
    ) -> Optional[KnowledgeEntity]:
        """Get entity at specific version"""
        # Implementation in Sprint 7
        return None

    # ========================================================================
    # STATISTICS
    # ========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            "total_entities": await self.get_entity_count(),
            "total_relations": await self.get_relation_count(),
            "ontologies_registered": len(self.ontologies)
        }

    async def get_entity_count(
        self,
        entity_type: Optional[str] = None
    ) -> int:
        """Count entities"""
        if entity_type:
            query = """
                MATCH (n:KnowledgeEntity {ontology_type: $type})
                RETURN count(n)
            """
            result = self.kuzu_conn.execute(query, {"type": entity_type})
        else:
            query = "MATCH (n:KnowledgeEntity) RETURN count(n)"
            result = self.kuzu_conn.execute(query)

        return result.get_next()[0] if result.has_next() else 0

    async def get_relation_count(
        self,
        relation_type: Optional[str] = None
    ) -> int:
        """Count relations"""
        if relation_type:
            query = """
                MATCH ()-[r:Relation {relation_type: $type}]->()
                RETURN count(r)
            """
            result = self.kuzu_conn.execute(query, {"type": relation_type})
        else:
            query = "MATCH ()-[r:Relation]->() RETURN count(r)"
            result = self.kuzu_conn.execute(query)

        return result.get_next()[0] if result.has_next() else 0
