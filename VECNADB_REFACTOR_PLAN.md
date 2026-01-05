# VecnaDB Refactoring Plan

## Executive Summary

This document outlines the comprehensive refactoring of Cognee into **VecnaDB**, an ontology-native hybrid vector-graph AI database. The refactor transforms Cognee's memory-oriented architecture into a truth-oriented knowledge system where **meaning lives in vectors** and **truth lives in structure**.

---

## Phase 1: Foundation and Core Architecture

### 1.1 Project Renaming and Structure

**Current State:**
- Project name: `cognee`
- Module terminology: "memory", "chunks", "cognify"
- Focus: Memory storage and recall

**Target State:**
- Project name: `vecnadb`
- Module terminology: "ontology", "graph", "vector", "reasoning", "knowledge"
- Focus: Knowledge validation and truth representation

**Actions:**
```
cognee/                          →  vecnadb/
├── api/v1/                      →  vecnadb/api/v1/
├── modules/cognify/             →  vecnadb/modules/knowledge/
├── modules/memify/              →  vecnadb/modules/reasoning/
├── modules/chunking/            →  vecnadb/modules/ingestion/
├── modules/retrieval/           →  vecnadb/modules/query/
├── modules/ontology/            →  vecnadb/modules/ontology/ (enhanced)
├── modules/graph/               →  vecnadb/modules/graph/ (core)
├── infrastructure/engine/       →  vecnadb/infrastructure/storage/
└── NEW: vecnadb/modules/vector/
```

**Key Files to Rename:**
- `cognify.py` → `ingest.py` or `build_knowledge.py`
- `add.py` → `insert.py` or `add_entity.py`
- `base_config.py` → `vecnadb_config.py`

---

### 1.2 Core Data Model Refactor

#### 1.2.1 Enhanced DataPoint (Knowledge Entity)

**Current DataPoint:**
```python
class DataPoint(BaseModel):
    id: UUID
    created_at: int
    updated_at: int
    ontology_valid: bool
    version: int
    topological_rank: Optional[int]
    metadata: Optional[MetaData]
    type: str
```

**VecnaDB KnowledgeEntity:**
```python
class KnowledgeEntity(BaseModel):
    # Identity
    id: UUID
    type: str  # MUST match ontology entity type

    # Ontology enforcement
    ontology_id: UUID  # Reference to ontology version
    ontology_type: str  # Entity type from ontology
    ontology_valid: bool  # Validation status
    validation_errors: Optional[List[str]]

    # Temporal tracking
    created_at: datetime
    updated_at: datetime
    version: int
    supersedes: Optional[UUID]  # Previous version

    # Graph properties
    graph_node_id: str  # Unique graph identifier
    topological_rank: Optional[int]

    # Vector properties
    embeddings: List[EmbeddingRecord]  # MUST have at least one

    # Metadata
    metadata: Dict[str, Any]
    provenance: ProvenanceRecord
```

**EmbeddingRecord:**
```python
class EmbeddingRecord(BaseModel):
    id: UUID
    entity_id: UUID  # MUST reference a KnowledgeEntity
    embedding_type: EmbeddingType  # CONTENT, SUMMARY, ROLE, etc.
    vector: List[float]
    model: str  # e.g., "text-embedding-3-small"
    model_version: str
    created_at: datetime
    dimensions: int
    metadata: Dict[str, Any]
```

**ProvenanceRecord:**
```python
class ProvenanceRecord(BaseModel):
    source_document: Optional[UUID]
    extraction_method: str  # "llm", "manual", "imported"
    extraction_model: Optional[str]
    confidence_score: Optional[float]
    created_by: Optional[str]
    modified_by: Optional[str]
```

#### 1.2.2 Dual Representation Enforcement

**Rule:** Every KnowledgeEntity MUST:
1. Exist as a typed graph node
2. Have at least one vector embedding

**Implementation:**
```python
# In storage layer
async def add_entity(entity: KnowledgeEntity):
    # Validate ontology compliance
    await ontology_validator.validate(entity)

    # Enforce dual representation
    if not entity.embeddings:
        raise ValueError("Entity must have at least one embedding")

    # Atomic transaction
    async with transaction():
        # 1. Add to graph
        await graph_db.add_node(
            node_id=entity.graph_node_id,
            entity=entity,
            properties=entity.model_dump()
        )

        # 2. Add embeddings to vector index
        for embedding in entity.embeddings:
            await vector_db.index_embedding(
                collection=entity.ontology_type,
                entity_id=entity.id,
                embedding=embedding
            )
```

---

### 1.3 Ontology System (Core of VecnaDB)

#### 1.3.1 Ontology Definition Model

**OntologySchema:**
```python
class OntologySchema(BaseModel):
    id: UUID
    name: str
    version: str
    created_at: datetime

    # Entity type definitions
    entity_types: Dict[str, EntityTypeDefinition]

    # Relationship type definitions
    relation_types: Dict[str, RelationTypeDefinition]

    # Inheritance rules
    inheritance_graph: Dict[str, List[str]]  # child → parents

    # Constraints
    global_constraints: List[Constraint]
```

**EntityTypeDefinition:**
```python
class EntityTypeDefinition(BaseModel):
    name: str
    description: str
    properties: Dict[str, PropertyDefinition]
    required_properties: List[str]
    inherits_from: Optional[List[str]]
    constraints: List[Constraint]
    embedding_requirements: EmbeddingRequirements
```

**RelationTypeDefinition:**
```python
class RelationTypeDefinition(BaseModel):
    name: str
    description: str

    # Type constraints
    allowed_source_types: List[str]
    allowed_target_types: List[str]

    # Cardinality
    source_cardinality: Cardinality  # ONE, MANY
    target_cardinality: Cardinality

    # Directionality
    is_directed: bool
    symmetric: bool  # If A→B, then B→A
    transitive: bool  # If A→B and B→C, then A→C

    # Properties
    properties: Dict[str, PropertyDefinition]
    constraints: List[Constraint]
```

**PropertyDefinition:**
```python
class PropertyDefinition(BaseModel):
    name: str
    type: PropertyType  # STRING, INT, FLOAT, BOOL, UUID, DATETIME, LIST, DICT
    required: bool
    default: Optional[Any]
    constraints: List[Constraint]
    indexed: bool
    embeddable: bool  # Include in vector embedding
```

**Constraint:**
```python
class Constraint(BaseModel):
    type: ConstraintType  # REGEX, RANGE, ENUM, UNIQUE, REFERENCE, CUSTOM
    parameters: Dict[str, Any]
    error_message: str
```

#### 1.3.2 Ontology Validation Engine

**OntologyValidator:**
```python
class OntologyValidator:
    def __init__(self, ontology: OntologySchema):
        self.ontology = ontology

    async def validate_entity(self, entity: KnowledgeEntity) -> ValidationResult:
        """
        Validates entity against ontology:
        1. Type exists in ontology
        2. Required properties present
        3. Property types match
        4. Constraints satisfied
        5. Inheritance rules followed
        """
        errors = []

        # Check type exists
        if entity.ontology_type not in self.ontology.entity_types:
            errors.append(f"Unknown entity type: {entity.ontology_type}")
            return ValidationResult(valid=False, errors=errors)

        type_def = self.ontology.entity_types[entity.ontology_type]

        # Check required properties
        for prop_name in type_def.required_properties:
            if not hasattr(entity, prop_name):
                errors.append(f"Missing required property: {prop_name}")

        # Check property types and constraints
        for prop_name, prop_def in type_def.properties.items():
            if hasattr(entity, prop_name):
                value = getattr(entity, prop_name)
                if not self._validate_property(value, prop_def):
                    errors.append(f"Invalid property {prop_name}: {value}")

        # Check entity-level constraints
        for constraint in type_def.constraints:
            if not self._validate_constraint(entity, constraint):
                errors.append(constraint.error_message)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )

    async def validate_relation(
        self,
        source: KnowledgeEntity,
        relation_type: str,
        target: KnowledgeEntity
    ) -> ValidationResult:
        """
        Validates relation against ontology:
        1. Relation type exists
        2. Source type allowed
        3. Target type allowed
        4. Cardinality constraints
        5. Directionality rules
        """
        errors = []

        # Check relation type exists
        if relation_type not in self.ontology.relation_types:
            errors.append(f"Unknown relation type: {relation_type}")
            return ValidationResult(valid=False, errors=errors)

        rel_def = self.ontology.relation_types[relation_type]

        # Check source type
        if source.ontology_type not in rel_def.allowed_source_types:
            errors.append(
                f"Invalid source type {source.ontology_type} "
                f"for relation {relation_type}"
            )

        # Check target type
        if target.ontology_type not in rel_def.allowed_target_types:
            errors.append(
                f"Invalid target type {target.ontology_type} "
                f"for relation {relation_type}"
            )

        # TODO: Check cardinality constraints (requires graph query)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )
```

#### 1.3.3 Built-in Core Ontology

**VecnaDB will ship with a core ontology:**

```yaml
# vecnadb/ontologies/core.yaml
name: "VecnaDB Core Ontology"
version: "1.0.0"

entity_types:
  Entity:
    description: "Base entity type for all knowledge entities"
    properties:
      name:
        type: STRING
        required: true
        embeddable: true
      description:
        type: STRING
        required: false
        embeddable: true

  Concept:
    inherits_from: [Entity]
    description: "Abstract concept or idea"
    properties:
      definition:
        type: STRING
        required: true
        embeddable: true

  Document:
    inherits_from: [Entity]
    description: "Source document"
    properties:
      content:
        type: STRING
        required: true
        embeddable: true
      content_type:
        type: STRING
        required: true
      source_uri:
        type: STRING

  Person:
    inherits_from: [Entity]
    description: "Individual person"
    properties:
      full_name:
        type: STRING
        required: true
        embeddable: true

  Organization:
    inherits_from: [Entity]
    description: "Organization or institution"

  Event:
    inherits_from: [Entity]
    description: "Temporal event"
    properties:
      timestamp:
        type: DATETIME
        required: true

relation_types:
  IS_A:
    description: "Type/subtype relationship"
    allowed_source_types: [Entity]
    allowed_target_types: [Entity]
    is_directed: true
    transitive: true

  PART_OF:
    description: "Part-whole relationship"
    allowed_source_types: [Entity]
    allowed_target_types: [Entity]
    is_directed: true
    transitive: true

  RELATED_TO:
    description: "Generic semantic relationship"
    allowed_source_types: [Entity]
    allowed_target_types: [Entity]
    is_directed: false
    symmetric: true

  DEFINED_IN:
    description: "Entity defined in document"
    allowed_source_types: [Entity, Concept]
    allowed_target_types: [Document]
    is_directed: true

  MENTIONS:
    description: "Document mentions entity"
    allowed_source_types: [Document]
    allowed_target_types: [Entity]
    is_directed: true
```

---

## Phase 2: Knowledge Model Transformation

### 2.1 Replace Memory Chunks with Entity-Relation Model

**Current Model:**
```python
# Flat chunk storage
DocumentChunk:
  - text: str
  - chunk_size: int
  - chunk_index: int
  - is_part_of: Document
  - contains: List[Union[Entity, Event, Edge]]  # Weak typing
```

**VecnaDB Model:**
```python
# Contextual subgraph
TextSegment(KnowledgeEntity):
  ontology_type: "TextSegment"
  content: str
  position_in_document: int

  # Relations (enforced by ontology):
  # - PART_OF → Document
  # - MENTIONS → List[Entity]
  # - PRECEDES → TextSegment (order)

  # Multiple embeddings:
  embeddings: [
    EmbeddingRecord(type=CONTENT, vector=[...]),
    EmbeddingRecord(type=SUMMARY, vector=[...])
  ]
```

**Context Representation:**
```python
# OLD: List of chunks
context = [chunk1, chunk2, chunk3]

# NEW: Bounded subgraph
class SubgraphContext:
    central_nodes: List[UUID]  # Query-relevant entities
    edges: List[Relation]
    metadata: Dict[str, Any]

    def to_text(self) -> str:
        """Convert subgraph to LLM-readable context"""
        # Generate narrative from graph structure

    def to_graph(self) -> CogneeGraph:
        """Return as graph object"""
```

### 2.2 Context-as-Subgraph Implementation

**SubgraphExtractor:**
```python
class SubgraphExtractor:
    async def extract_context(
        self,
        query: str,
        query_vector: List[float],
        ontology: OntologySchema,
        max_nodes: int = 50,
        max_depth: int = 3
    ) -> SubgraphContext:
        """
        Extract ontology-constrained subgraph for query

        Process:
        1. Vector search for semantically similar entities
        2. Filter by ontology type constraints
        3. Expand graph by traversing typed relations
        4. Rank nodes by combined graph+vector score
        5. Prune to bounded subgraph
        """

        # Step 1: Semantic retrieval
        candidates = await self._vector_search(
            query_vector=query_vector,
            top_k=100
        )

        # Step 2: Ontology filtering
        valid_candidates = await self._ontology_filter(
            candidates=candidates,
            ontology=ontology
        )

        # Step 3: Graph expansion
        subgraph = await self._expand_subgraph(
            seed_nodes=valid_candidates[:10],
            max_depth=max_depth,
            relation_types=ontology.relation_types.keys()
        )

        # Step 4: Hybrid ranking
        ranked_nodes = await self._rank_nodes(
            subgraph=subgraph,
            query_vector=query_vector,
            graph_centrality_weight=0.3,
            vector_similarity_weight=0.7
        )

        # Step 5: Bounded extraction
        return SubgraphContext(
            central_nodes=ranked_nodes[:max_nodes],
            edges=await self._extract_edges(ranked_nodes[:max_nodes]),
            metadata={
                "query": query,
                "extraction_time": datetime.utcnow(),
                "ontology_version": ontology.version
            }
        )
```

---

## Phase 3: Storage Architecture

### 3.1 Unified Storage Interface

**Current State:**
- Separate `VectorDBInterface` and `GraphDBInterface`
- Dual writes to both systems
- Potential consistency issues

**VecnaDB State:**
```python
class VecnaDBStorageInterface(ABC):
    """
    Unified interface for hybrid vector-graph storage
    """

    # Entity operations
    @abstractmethod
    async def add_entity(self, entity: KnowledgeEntity) -> None:
        """Add entity with automatic dual representation"""

    @abstractmethod
    async def add_entities(self, entities: List[KnowledgeEntity]) -> None:
        """Batch entity insertion"""

    @abstractmethod
    async def get_entity(self, entity_id: UUID) -> KnowledgeEntity:
        """Retrieve entity by ID"""

    @abstractmethod
    async def update_entity(self, entity: KnowledgeEntity) -> None:
        """Update entity (creates new version)"""

    @abstractmethod
    async def delete_entity(self, entity_id: UUID) -> None:
        """Soft delete entity"""

    # Relation operations
    @abstractmethod
    async def add_relation(
        self,
        source_id: UUID,
        relation_type: str,
        target_id: UUID,
        properties: Dict[str, Any] = None
    ) -> UUID:
        """Add typed relation between entities"""

    @abstractmethod
    async def get_relations(
        self,
        entity_id: UUID,
        relation_type: Optional[str] = None,
        direction: Direction = Direction.BOTH
    ) -> List[Relation]:
        """Get relations for entity"""

    # Hybrid query operations
    @abstractmethod
    async def hybrid_search(
        self,
        query: HybridQuery
    ) -> SearchResult:
        """
        Execute hybrid vector-graph query

        HybridQuery supports:
        - Vector similarity constraints
        - Graph structure constraints
        - Ontology type filters
        - Combined ranking
        """

    # Subgraph operations
    @abstractmethod
    async def extract_subgraph(
        self,
        seed_nodes: List[UUID],
        max_depth: int = 3,
        filters: SubgraphFilters = None
    ) -> Subgraph:
        """Extract bounded subgraph around seed nodes"""

    # Ontology operations
    @abstractmethod
    async def register_ontology(self, ontology: OntologySchema) -> None:
        """Register new ontology version"""

    @abstractmethod
    async def get_ontology(self, ontology_id: UUID) -> OntologySchema:
        """Retrieve ontology definition"""

    # Versioning operations
    @abstractmethod
    async def get_entity_history(self, entity_id: UUID) -> List[KnowledgeEntity]:
        """Get all versions of entity"""

    # Embedding operations
    @abstractmethod
    async def update_embeddings(
        self,
        entity_id: UUID,
        embeddings: List[EmbeddingRecord]
    ) -> None:
        """Update entity embeddings (supports re-embedding)"""
```

### 3.2 Adapter Pattern for Storage Backends

**Implement adapters for existing backends:**

```python
# vecnadb/infrastructure/storage/adapters/

class LanceDBKuzuAdapter(VecnaDBStorageInterface):
    """Hybrid adapter using LanceDB + Kuzu"""

    def __init__(self, lance_db: LanceDB, kuzu_db: Kuzu):
        self.vector_store = lance_db
        self.graph_store = kuzu_db
        self.ontology_validator = None

    async def add_entity(self, entity: KnowledgeEntity):
        # Validate ontology
        validation = await self.ontology_validator.validate_entity(entity)
        if not validation.valid:
            raise OntologyValidationError(validation.errors)

        # Atomic dual write
        async with transaction():
            # Graph
            await self.graph_store.add_node(
                node_id=str(entity.id),
                label=entity.ontology_type,
                properties=entity.model_dump(exclude={'embeddings'})
            )

            # Vector
            for embedding in entity.embeddings:
                await self.vector_store.create_data_points(
                    collection_name=entity.ontology_type,
                    data_points=[{
                        'entity_id': str(entity.id),
                        'vector': embedding.vector,
                        'embedding_type': embedding.embedding_type,
                        'metadata': entity.model_dump()
                    }]
                )

class Neo4jAdapter(VecnaDBStorageInterface):
    """Adapter for Neo4j with vector index"""
    # Neo4j 5.13+ has native vector search
    pass

class NeptuneAnalyticsAdapter(VecnaDBStorageInterface):
    """Adapter for AWS Neptune Analytics (native hybrid)"""
    # Neptune Analytics natively supports hybrid queries
    pass
```

### 3.3 Migration from Dual Storage to Unified

**DataPoint → KnowledgeEntity Migration:**

```python
# vecnadb/infrastructure/storage/migrations/cognee_to_vecnadb.py

class CogneeToVecnaDBMigrator:
    async def migrate_data_point(self, data_point: DataPoint) -> KnowledgeEntity:
        """
        Convert Cognee DataPoint to VecnaDB KnowledgeEntity
        """

        # Extract existing embeddings from vector DB
        embeddings = await self._extract_embeddings(data_point.id)

        # Determine ontology type from DataPoint.type
        ontology_type = self._map_type(data_point.type)

        # Create KnowledgeEntity
        entity = KnowledgeEntity(
            id=data_point.id,
            type=ontology_type,
            ontology_id=self.core_ontology_id,
            ontology_type=ontology_type,
            ontology_valid=data_point.ontology_valid,
            created_at=datetime.fromtimestamp(data_point.created_at),
            updated_at=datetime.fromtimestamp(data_point.updated_at),
            version=data_point.version,
            graph_node_id=str(data_point.id),
            topological_rank=data_point.topological_rank,
            embeddings=embeddings,
            metadata=data_point.metadata or {},
            provenance=ProvenanceRecord(
                source_document=None,
                extraction_method="migration",
                created_by="cognee_migrator"
            )
        )

        return entity

    async def migrate_all(self):
        """
        Full migration process:
        1. Load all DataPoints from Cognee
        2. Convert to KnowledgeEntities
        3. Validate against core ontology
        4. Insert into VecnaDB storage
        """
        # Implementation
        pass
```

---

## Phase 4: Query System

### 4.1 Hybrid Query Model

**HybridQuery:**
```python
class HybridQuery(BaseModel):
    # Query input
    query_text: str
    query_vector: Optional[List[float]]  # Pre-computed or will be embedded

    # Vector constraints
    vector_search: VectorSearchConfig = VectorSearchConfig(
        enabled=True,
        top_k=50,
        similarity_threshold=0.7,
        embedding_types=[EmbeddingType.CONTENT]
    )

    # Graph constraints
    graph_traversal: GraphTraversalConfig = GraphTraversalConfig(
        enabled=True,
        max_depth=3,
        relation_types=None,  # None = all types
        direction=Direction.BOTH
    )

    # Ontology constraints
    ontology_filter: OntologyFilter = OntologyFilter(
        entity_types=None,  # None = all types
        exclude_types=[],
        relation_types=None
    )

    # Ranking
    ranking: RankingConfig = RankingConfig(
        vector_weight=0.7,
        graph_weight=0.3,
        rank_by=[RankingMetric.SEMANTIC_SIMILARITY, RankingMetric.CENTRALITY]
    )

    # Output
    output: OutputConfig = OutputConfig(
        format=OutputFormat.SUBGRAPH,  # SUBGRAPH, ENTITIES, TEXT
        max_results=20,
        include_paths=True,
        include_explanations=True
    )
```

**SearchResult:**
```python
class SearchResult(BaseModel):
    query: HybridQuery
    results: List[SearchResultItem]
    subgraph: Optional[SubgraphContext]
    execution_metadata: ExecutionMetadata

class SearchResultItem(BaseModel):
    entity: KnowledgeEntity
    score: float
    similarity_score: Optional[float]
    graph_score: Optional[float]
    path_from_query: Optional[List[Relation]]
    explanation: str  # Human-readable explanation

class ExecutionMetadata(BaseModel):
    execution_time_ms: float
    vector_search_time_ms: float
    graph_traversal_time_ms: float
    ranking_time_ms: float
    total_candidates: int
    ontology_filtered: int
    final_results: int
```

### 4.2 Query Executor

**HybridQueryExecutor:**
```python
class HybridQueryExecutor:
    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema
    ):
        self.storage = storage
        self.ontology = ontology

    async def execute(self, query: HybridQuery) -> SearchResult:
        """
        Execute hybrid query with ontology constraints
        """
        start_time = time.time()
        metadata = {}

        # Step 1: Embed query if needed
        if not query.query_vector:
            query.query_vector = await self._embed_query(query.query_text)

        # Step 2: Vector search (if enabled)
        vector_candidates = []
        if query.vector_search.enabled:
            vector_start = time.time()
            vector_candidates = await self._vector_search(query)
            metadata['vector_search_time_ms'] = (time.time() - vector_start) * 1000

        # Step 3: Ontology filtering
        filtered_candidates = await self._ontology_filter(
            candidates=vector_candidates,
            filter_config=query.ontology_filter
        )
        metadata['ontology_filtered'] = len(vector_candidates) - len(filtered_candidates)

        # Step 4: Graph expansion (if enabled)
        subgraph = None
        if query.graph_traversal.enabled:
            graph_start = time.time()
            subgraph = await self._expand_subgraph(
                seed_nodes=[c.entity.id for c in filtered_candidates],
                traversal_config=query.graph_traversal,
                ontology_filter=query.ontology_filter
            )
            metadata['graph_traversal_time_ms'] = (time.time() - graph_start) * 1000

        # Step 5: Hybrid ranking
        rank_start = time.time()
        ranked_results = await self._rank_results(
            candidates=filtered_candidates,
            subgraph=subgraph,
            ranking_config=query.ranking,
            query_vector=query.query_vector
        )
        metadata['ranking_time_ms'] = (time.time() - rank_start) * 1000

        # Step 6: Generate explanations
        if query.output.include_explanations:
            ranked_results = await self._add_explanations(ranked_results, query)

        # Step 7: Format output
        final_results = ranked_results[:query.output.max_results]

        return SearchResult(
            query=query,
            results=final_results,
            subgraph=subgraph if query.output.format == OutputFormat.SUBGRAPH else None,
            execution_metadata=ExecutionMetadata(
                execution_time_ms=(time.time() - start_time) * 1000,
                total_candidates=len(vector_candidates),
                final_results=len(final_results),
                **metadata
            )
        )

    async def _ontology_filter(
        self,
        candidates: List[SearchResultItem],
        filter_config: OntologyFilter
    ) -> List[SearchResultItem]:
        """
        Filter candidates by ontology constraints
        """
        filtered = []

        for candidate in candidates:
            entity = candidate.entity

            # Type filter
            if filter_config.entity_types:
                if entity.ontology_type not in filter_config.entity_types:
                    continue

            # Exclusion filter
            if entity.ontology_type in filter_config.exclude_types:
                continue

            # Ontology validity
            if not entity.ontology_valid:
                continue

            filtered.append(candidate)

        return filtered

    async def _add_explanations(
        self,
        results: List[SearchResultItem],
        query: HybridQuery
    ) -> List[SearchResultItem]:
        """
        Add human-readable explanations to results

        Explanation includes:
        - Why this entity was retrieved (vector similarity? graph proximity?)
        - What path led to it
        - What ontology rules applied
        """
        for result in results:
            explanation_parts = []

            # Vector similarity explanation
            if result.similarity_score:
                explanation_parts.append(
                    f"Semantically similar to query (score: {result.similarity_score:.2f})"
                )

            # Graph proximity explanation
            if result.path_from_query:
                path_str = " → ".join([
                    f"{rel.relation_type}" for rel in result.path_from_query
                ])
                explanation_parts.append(
                    f"Connected via: {path_str}"
                )

            # Ontology explanation
            explanation_parts.append(
                f"Entity type: {result.entity.ontology_type}"
            )

            result.explanation = "; ".join(explanation_parts)

        return results
```

---

## Phase 5: Reasoning Layer

### 5.1 Deterministic Reasoning (Graph-Based)

**GraphReasoner:**
```python
class GraphReasoner:
    """
    Deterministic reasoning using graph structure and ontology
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema
    ):
        self.storage = storage
        self.ontology = ontology

    async def resolve_inheritance(
        self,
        entity: KnowledgeEntity
    ) -> List[str]:
        """
        Resolve entity type inheritance chain

        Example:
        - Person IS_A Entity
        - Scientist IS_A Person
        - Scientist inherits properties from Person and Entity
        """
        type_chain = [entity.ontology_type]
        current_type = entity.ontology_type

        while current_type in self.ontology.inheritance_graph:
            parents = self.ontology.inheritance_graph[current_type]
            type_chain.extend(parents)
            current_type = parents[0] if parents else None

        return type_chain

    async def check_constraint(
        self,
        entity: KnowledgeEntity,
        constraint: Constraint
    ) -> bool:
        """
        Deterministically check if entity satisfies constraint
        """
        # Implementation based on constraint type
        pass

    async def infer_relations(
        self,
        entity: KnowledgeEntity
    ) -> List[Relation]:
        """
        Infer implicit relations from transitive/symmetric rules

        Example:
        - If A PART_OF B and B PART_OF C, infer A PART_OF C (transitivity)
        - If A RELATED_TO B, infer B RELATED_TO A (symmetry)
        """
        inferred = []

        # Get explicit relations
        explicit = await self.storage.get_relations(entity.id)

        # Apply transitive rules
        for relation_type, rel_def in self.ontology.relation_types.items():
            if rel_def.transitive:
                inferred.extend(
                    await self._apply_transitivity(entity.id, relation_type)
                )

        # Apply symmetric rules
        for relation in explicit:
            rel_def = self.ontology.relation_types[relation.relation_type]
            if rel_def.symmetric:
                inferred.append(
                    Relation(
                        source_id=relation.target_id,
                        relation_type=relation.relation_type,
                        target_id=relation.source_id,
                        properties=relation.properties,
                        inferred=True
                    )
                )

        return inferred

    async def extract_subgraph_with_rules(
        self,
        seed_nodes: List[UUID],
        max_depth: int = 3
    ) -> Subgraph:
        """
        Extract subgraph and apply reasoning rules to enrich it
        """
        # Extract base subgraph
        subgraph = await self.storage.extract_subgraph(
            seed_nodes=seed_nodes,
            max_depth=max_depth
        )

        # Apply inference rules
        for node in subgraph.nodes:
            inferred_relations = await self.infer_relations(node)
            subgraph.edges.extend(inferred_relations)

        return subgraph
```

### 5.2 Probabilistic Reasoning (Vector-Based)

**VectorReasoner:**
```python
class VectorReasoner:
    """
    Probabilistic reasoning using vector similarity

    CRITICAL: Vectors are ADVISORY only, never authoritative
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        embedding_service: EmbeddingService
    ):
        self.storage = storage
        self.embedding_service = embedding_service

    async def rank_by_similarity(
        self,
        entities: List[KnowledgeEntity],
        query_vector: List[float],
        embedding_type: EmbeddingType = EmbeddingType.CONTENT
    ) -> List[Tuple[KnowledgeEntity, float]]:
        """
        Rank entities by vector similarity (for recall/ranking only)
        """
        ranked = []

        for entity in entities:
            # Get relevant embedding
            embedding = next(
                (e for e in entity.embeddings if e.embedding_type == embedding_type),
                None
            )

            if not embedding:
                continue

            # Calculate similarity
            similarity = self._cosine_similarity(query_vector, embedding.vector)
            ranked.append((entity, similarity))

        # Sort by similarity descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    async def expand_semantic_neighbors(
        self,
        entity: KnowledgeEntity,
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[KnowledgeEntity]:
        """
        Find semantically similar entities (for expansion/recall)

        Use case: Expand query with semantically related but not
        graph-connected entities
        """
        # Get entity embedding
        embedding = entity.embeddings[0]

        # Vector search
        similar = await self.storage.hybrid_search(
            HybridQuery(
                query_text="",
                query_vector=embedding.vector,
                vector_search=VectorSearchConfig(
                    enabled=True,
                    top_k=top_k,
                    similarity_threshold=threshold
                ),
                graph_traversal=GraphTraversalConfig(enabled=False)
            )
        )

        return [r.entity for r in similar.results]

    async def suggest_relations(
        self,
        source: KnowledgeEntity,
        target: KnowledgeEntity
    ) -> List[Tuple[str, float]]:
        """
        Suggest possible relation types based on vector similarity

        CRITICAL: These are SUGGESTIONS only, must be validated by graph
        """
        # Create triplet text for each possible relation type
        suggestions = []

        for relation_type in self.ontology.relation_types:
            triplet_text = f"{source.name} {relation_type} {target.name}"
            triplet_vector = await self.embedding_service.embed(triplet_text)

            # Compare against known triplets of this type
            score = await self._score_triplet_likelihood(
                triplet_vector,
                relation_type
            )

            suggestions.append((relation_type, score))

        # Sort by likelihood
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return suggestions
```

### 5.3 Hybrid Reasoning Orchestrator

**ReasoningEngine:**
```python
class ReasoningEngine:
    """
    Orchestrates deterministic and probabilistic reasoning
    """

    def __init__(
        self,
        graph_reasoner: GraphReasoner,
        vector_reasoner: VectorReasoner
    ):
        self.graph = graph_reasoner
        self.vector = vector_reasoner

    async def answer_query_with_reasoning(
        self,
        query: str,
        reasoning_depth: int = 2
    ) -> ReasonedAnswer:
        """
        Answer query using multi-step reasoning

        Process:
        1. Vector search for recall (probabilistic)
        2. Graph expansion for structure (deterministic)
        3. Inference for implicit knowledge (deterministic)
        4. Semantic expansion for coverage (probabilistic)
        5. Ranking for relevance (probabilistic)
        """

        # Step 1: Initial vector search
        query_vector = await self.vector.embedding_service.embed(query)
        initial_results = await self.storage.hybrid_search(
            HybridQuery(query_text=query, query_vector=query_vector)
        )

        # Step 2: Graph-based reasoning
        seed_nodes = [r.entity.id for r in initial_results.results[:5]]
        reasoned_subgraph = await self.graph.extract_subgraph_with_rules(
            seed_nodes=seed_nodes,
            max_depth=reasoning_depth
        )

        # Step 3: Semantic expansion
        expanded_entities = []
        for node in reasoned_subgraph.nodes[:3]:
            neighbors = await self.vector.expand_semantic_neighbors(
                entity=node,
                top_k=5,
                threshold=0.75
            )
            expanded_entities.extend(neighbors)

        # Step 4: Combine graph and vector evidence
        all_entities = list(reasoned_subgraph.nodes) + expanded_entities
        ranked = await self.vector.rank_by_similarity(
            entities=all_entities,
            query_vector=query_vector
        )

        # Step 5: Generate answer with provenance
        return ReasonedAnswer(
            query=query,
            answer=await self._generate_answer(ranked[:10]),
            entities_used=[e for e, _ in ranked[:10]],
            reasoning_subgraph=reasoned_subgraph,
            provenance=self._generate_provenance(ranked, reasoned_subgraph)
        )
```

---

## Phase 6: Ontology-Guided RAG

### 6.1 RAG Pipeline Refactor

**Current RAG (Cognee):**
```python
# 1. Vector search on chunks
# 2. Return top-k chunks
# 3. Feed to LLM
# Problem: No structure, no validation, no explanation
```

**VecnaDB Ontology-Guided RAG:**
```python
class OntologyGuidedRAG:
    """
    RAG pipeline that enforces ontology constraints and provides explainability
    """

    async def retrieve_and_generate(
        self,
        query: str,
        ontology_constraints: OntologyFilter = None,
        max_context_tokens: int = 8000
    ) -> RAGResponse:
        """
        Ontology-guided RAG with full explainability
        """

        # Step 1: Hybrid query with ontology constraints
        search_result = await self.query_executor.execute(
            HybridQuery(
                query_text=query,
                ontology_filter=ontology_constraints or OntologyFilter(),
                output=OutputConfig(
                    format=OutputFormat.SUBGRAPH,
                    include_explanations=True
                )
            )
        )

        # Step 2: Convert subgraph to LLM context
        context = await self._subgraph_to_context(
            subgraph=search_result.subgraph,
            max_tokens=max_context_tokens
        )

        # Step 3: Validate context against ontology
        validated_context = await self._validate_context(
            context=context,
            ontology=self.ontology
        )

        # Step 4: Generate answer
        llm_response = await self.llm.generate(
            prompt=self._build_prompt(query, validated_context)
        )

        # Step 5: Ground answer in graph
        grounded_answer = await self._ground_answer(
            answer=llm_response,
            subgraph=search_result.subgraph
        )

        return RAGResponse(
            query=query,
            answer=grounded_answer.text,
            context_subgraph=search_result.subgraph,
            entities_cited=grounded_answer.cited_entities,
            relations_cited=grounded_answer.cited_relations,
            ontology_constraints_applied=ontology_constraints,
            execution_metadata=search_result.execution_metadata,
            explainability=self._generate_explainability(
                search_result, grounded_answer
            )
        )

    async def _validate_context(
        self,
        context: SubgraphContext,
        ontology: OntologySchema
    ) -> SubgraphContext:
        """
        Remove semantically relevant but ontologically invalid nodes

        Example: Query about "programming languages"
        - Returns "Python" (entity type: Language) ✓
        - Returns "Monty Python" (entity type: ComedyGroup) ✗
          (semantically similar but ontologically invalid)
        """
        valid_nodes = []

        for node in context.central_nodes:
            entity = await self.storage.get_entity(node)

            # Check ontology validity
            if not entity.ontology_valid:
                continue

            # Check if entity type makes sense for query context
            # (This could use LLM-based validation or rule-based)
            if await self._is_contextually_valid(entity, context):
                valid_nodes.append(node)

        return SubgraphContext(
            central_nodes=valid_nodes,
            edges=await self._filter_edges(valid_nodes, context.edges),
            metadata=context.metadata
        )

    async def _ground_answer(
        self,
        answer: str,
        subgraph: SubgraphContext
    ) -> GroundedAnswer:
        """
        Ground LLM answer in graph structure

        - Extract entity mentions from answer
        - Link to graph nodes
        - Verify claims against graph structure
        """
        # Use NER or LLM to extract entities from answer
        mentioned_entities = await self._extract_entities(answer)

        # Map to graph nodes
        cited_entities = []
        for mention in mentioned_entities:
            node = await self._resolve_entity_mention(mention, subgraph)
            if node:
                cited_entities.append(node)

        # Extract relational claims
        cited_relations = await self._extract_relations(answer, subgraph)

        # Verify claims
        verified_answer = await self._verify_claims(
            answer, cited_entities, cited_relations, subgraph
        )

        return GroundedAnswer(
            text=verified_answer,
            cited_entities=cited_entities,
            cited_relations=cited_relations,
            verification_status=VerificationStatus.VERIFIED
        )
```

### 6.2 Hallucination Prevention

**OntologyHallucinationPreventer:**
```python
class OntologyHallucinationPreventer:
    """
    Prevents LLM hallucinations by enforcing ontology constraints
    """

    async def validate_llm_output(
        self,
        llm_output: str,
        context_subgraph: SubgraphContext,
        ontology: OntologySchema
    ) -> ValidationResult:
        """
        Validate LLM output against graph structure

        Checks:
        1. All mentioned entities exist in context subgraph
        2. All mentioned relations are valid per ontology
        3. All claims can be traced to graph structure
        """

        # Extract claims from LLM output
        claims = await self._extract_claims(llm_output)

        violations = []

        for claim in claims:
            # Check if claim is supported by subgraph
            if not await self._is_claim_supported(claim, context_subgraph):
                violations.append(
                    HallucinationViolation(
                        claim=claim,
                        reason="No support in knowledge graph",
                        severity=Severity.HIGH
                    )
                )

            # Check if claim violates ontology constraints
            if await self._violates_ontology(claim, ontology):
                violations.append(
                    HallucinationViolation(
                        claim=claim,
                        reason="Violates ontology constraints",
                        severity=Severity.CRITICAL
                    )
                )

        return ValidationResult(
            valid=len(violations) == 0,
            violations=violations
        )
```

---

## Phase 7: Versioning and Evolution

### 7.1 Entity Versioning

**Versioned entities:**
```python
class KnowledgeEntity(BaseModel):
    # ... existing fields ...
    version: int
    supersedes: Optional[UUID]  # Previous version
    superseded_by: Optional[UUID]  # Next version
    version_metadata: VersionMetadata

class VersionMetadata(BaseModel):
    change_type: ChangeType  # CREATED, UPDATED, MERGED, SPLIT
    changed_properties: List[str]
    change_reason: str
    changed_by: str
    changed_at: datetime
```

**Versioning operations:**
```python
class VersionManager:
    async def update_entity(
        self,
        entity_id: UUID,
        updates: Dict[str, Any],
        change_reason: str
    ) -> KnowledgeEntity:
        """
        Create new version of entity
        """
        # Get current version
        current = await self.storage.get_entity(entity_id)

        # Create new version
        new_version = current.model_copy(update=updates)
        new_version.id = uuid4()
        new_version.version = current.version + 1
        new_version.supersedes = current.id
        new_version.version_metadata = VersionMetadata(
            change_type=ChangeType.UPDATED,
            changed_properties=list(updates.keys()),
            change_reason=change_reason,
            changed_by=get_current_user(),
            changed_at=datetime.utcnow()
        )

        # Insert new version
        await self.storage.add_entity(new_version)

        # Update superseded pointer
        current.superseded_by = new_version.id
        await self.storage._update_metadata(current)

        return new_version

    async def get_entity_at_time(
        self,
        entity_id: UUID,
        timestamp: datetime
    ) -> Optional[KnowledgeEntity]:
        """
        Get entity version at specific timestamp
        """
        versions = await self.storage.get_entity_history(entity_id)

        # Find version valid at timestamp
        for version in sorted(versions, key=lambda v: v.created_at, reverse=True):
            if version.created_at <= timestamp:
                return version

        return None
```

### 7.2 Ontology Versioning

**Ontology evolution:**
```python
class OntologyEvolutionManager:
    async def evolve_ontology(
        self,
        current_ontology: OntologySchema,
        changes: OntologyChanges
    ) -> OntologySchema:
        """
        Create new ontology version with backward compatibility checks
        """
        # Validate changes don't break existing data
        validation = await self._validate_evolution(current_ontology, changes)

        if not validation.compatible:
            raise IncompatibleOntologyChangeError(validation.issues)

        # Create new version
        new_ontology = self._apply_changes(current_ontology, changes)
        new_ontology.id = uuid4()
        new_ontology.version = self._increment_version(current_ontology.version)
        new_ontology.supersedes = current_ontology.id

        # Register new version
        await self.storage.register_ontology(new_ontology)

        return new_ontology

    async def migrate_entities(
        self,
        from_ontology: UUID,
        to_ontology: UUID
    ) -> MigrationResult:
        """
        Migrate entities from one ontology version to another
        """
        # Get entities using old ontology
        entities = await self.storage.get_entities_by_ontology(from_ontology)

        # Load ontologies
        old_ont = await self.storage.get_ontology(from_ontology)
        new_ont = await self.storage.get_ontology(to_ontology)

        # Create migration plan
        migration_plan = await self._create_migration_plan(old_ont, new_ont)

        # Migrate entities
        results = []
        for entity in entities:
            result = await self._migrate_entity(entity, migration_plan, new_ont)
            results.append(result)

        return MigrationResult(
            total_entities=len(entities),
            successful=len([r for r in results if r.success]),
            failed=len([r for r in results if not r.success]),
            results=results
        )
```

---

## Phase 8: Observability and Auditability

### 8.1 Audit Log System

**AuditLog:**
```python
class AuditLog(BaseModel):
    id: UUID
    timestamp: datetime
    operation: Operation  # INSERT, UPDATE, DELETE, QUERY
    user: Optional[str]

    # What was accessed/modified
    entity_ids: List[UUID]
    relation_ids: List[UUID]

    # Operation details
    operation_type: str  # add_entity, hybrid_search, etc.
    parameters: Dict[str, Any]

    # Results
    success: bool
    error: Optional[str]

    # Ontology context
    ontology_id: UUID
    ontology_violations: List[str]

    # Query metadata (if applicable)
    query_metadata: Optional[QueryMetadata]

class QueryMetadata(BaseModel):
    query_text: str
    query_vector_model: str
    results_returned: int
    execution_time_ms: float
    vector_candidates: int
    graph_nodes_traversed: int
    ontology_filtered: int
```

**Audit Logger:**
```python
class AuditLogger:
    async def log_insertion(
        self,
        entity: KnowledgeEntity,
        success: bool,
        error: Optional[str] = None
    ):
        """Log entity insertion"""
        await self._write_log(
            AuditLog(
                id=uuid4(),
                timestamp=datetime.utcnow(),
                operation=Operation.INSERT,
                user=get_current_user(),
                entity_ids=[entity.id],
                relation_ids=[],
                operation_type="add_entity",
                parameters={"entity_type": entity.ontology_type},
                success=success,
                error=error,
                ontology_id=entity.ontology_id,
                ontology_violations=entity.validation_errors or []
            )
        )

    async def log_query(
        self,
        query: HybridQuery,
        result: SearchResult,
        success: bool
    ):
        """Log hybrid query execution"""
        await self._write_log(
            AuditLog(
                id=uuid4(),
                timestamp=datetime.utcnow(),
                operation=Operation.QUERY,
                user=get_current_user(),
                entity_ids=[r.entity.id for r in result.results],
                relation_ids=[],
                operation_type="hybrid_search",
                parameters=query.model_dump(),
                success=success,
                ontology_id=query.ontology_filter.ontology_id,
                query_metadata=QueryMetadata(
                    query_text=query.query_text,
                    query_vector_model=result.execution_metadata.vector_model,
                    results_returned=len(result.results),
                    execution_time_ms=result.execution_metadata.execution_time_ms,
                    vector_candidates=result.execution_metadata.total_candidates,
                    graph_nodes_traversed=result.execution_metadata.graph_nodes_traversed,
                    ontology_filtered=result.execution_metadata.ontology_filtered
                )
            )
        )

    async def reconstruct_result(
        self,
        log_id: UUID
    ) -> ReconstructedResult:
        """
        Reconstruct why a specific result was produced

        Enables full audit trail:
        - What query was executed
        - What entities were returned
        - Why they were returned (scores, paths)
        - What ontology rules applied
        """
        log = await self._get_log(log_id)

        return ReconstructedResult(
            query=log.query_metadata.query_text,
            entities=[await self.storage.get_entity(eid) for eid in log.entity_ids],
            reasoning=await self._reconstruct_reasoning(log),
            ontology_version=await self.storage.get_ontology(log.ontology_id)
        )
```

### 8.2 Observability Metrics

**Metrics to track:**
```python
class VecnaDBMetrics:
    # Performance metrics
    avg_query_time: Histogram
    vector_search_time: Histogram
    graph_traversal_time: Histogram
    ranking_time: Histogram

    # Volume metrics
    total_entities: Gauge
    total_relations: Gauge
    total_embeddings: Gauge

    # Quality metrics
    ontology_validation_failures: Counter
    ontology_validation_success_rate: Gauge

    # Query metrics
    queries_per_second: Gauge
    results_per_query: Histogram
    ontology_filter_rate: Histogram  # % of candidates filtered

    # Embedding metrics
    embeddings_generated: Counter
    embedding_dimensions: Histogram
    re_embeddings: Counter
```

---

## Phase 9: API Refactor

### 9.1 Public API Changes

**Current Cognee API:**
```python
# cognee/api/v1/
- add.py: add(data)
- cognify.py: cognify()
- search.py: search(query)
```

**VecnaDB API:**
```python
# vecnadb/api/v1/

# Entity management
- insert_entity(entity: KnowledgeEntity) -> UUID
- get_entity(entity_id: UUID) -> KnowledgeEntity
- update_entity(entity_id: UUID, updates: Dict) -> KnowledgeEntity
- delete_entity(entity_id: UUID) -> bool

# Relation management
- add_relation(source_id, relation_type, target_id, properties) -> UUID
- get_relations(entity_id, filters) -> List[Relation]

# Ontology management
- register_ontology(ontology: OntologySchema) -> UUID
- get_ontology(ontology_id: UUID) -> OntologySchema
- list_ontologies() -> List[OntologySchema]

# Query
- query(query: HybridQuery) -> SearchResult
- extract_subgraph(seed_nodes, config) -> Subgraph

# Ingestion (replaces cognify)
- ingest_document(document, ontology_id) -> List[UUID]  # Returns created entities

# RAG
- retrieve_and_generate(query, ontology_constraints) -> RAGResponse

# Reasoning
- reason(query, reasoning_depth) -> ReasonedAnswer

# Versioning
- get_entity_history(entity_id) -> List[KnowledgeEntity]
- get_entity_at_time(entity_id, timestamp) -> KnowledgeEntity

# Audit
- get_audit_logs(filters) -> List[AuditLog]
- reconstruct_result(log_id) -> ReconstructedResult
```

---

## Phase 10: Testing Strategy

### 10.1 Ontology Validation Tests

```python
# tests/test_ontology_validation.py

async def test_entity_type_validation():
    """Test that entities must match ontology types"""
    # Should succeed
    valid_entity = KnowledgeEntity(
        ontology_type="Person",
        ...
    )
    await storage.add_entity(valid_entity)

    # Should fail
    invalid_entity = KnowledgeEntity(
        ontology_type="InvalidType",
        ...
    )
    with pytest.raises(OntologyValidationError):
        await storage.add_entity(invalid_entity)

async def test_relation_type_constraints():
    """Test that relations enforce source/target type constraints"""
    person = create_entity(type="Person")
    document = create_entity(type="Document")
    concept = create_entity(type="Concept")

    # Valid: Person AUTHORED Document
    await storage.add_relation(person.id, "AUTHORED", document.id)

    # Invalid: Document AUTHORED Person (wrong direction)
    with pytest.raises(OntologyValidationError):
        await storage.add_relation(document.id, "AUTHORED", person.id)

async def test_dual_representation_enforcement():
    """Test that entities must have both graph node and vector embedding"""
    # Should fail: no embeddings
    entity_no_embeddings = KnowledgeEntity(
        ontology_type="Concept",
        embeddings=[],
        ...
    )
    with pytest.raises(ValueError, match="must have at least one embedding"):
        await storage.add_entity(entity_no_embeddings)
```

### 10.2 Hybrid Query Tests

```python
# tests/test_hybrid_query.py

async def test_ontology_constrained_search():
    """Test that ontology filters are applied correctly"""
    # Setup: Create entities of different types
    person = create_entity(type="Person", name="Alice")
    concept = create_entity(type="Concept", name="Alice in Wonderland")

    # Query with ontology filter
    result = await storage.hybrid_search(
        HybridQuery(
            query_text="Alice",
            ontology_filter=OntologyFilter(entity_types=["Person"])
        )
    )

    # Should only return Person, not Concept
    assert len(result.results) == 1
    assert result.results[0].entity.ontology_type == "Person"

async def test_explainability():
    """Test that query results include explanations"""
    result = await storage.hybrid_search(
        HybridQuery(
            query_text="programming languages",
            output=OutputConfig(include_explanations=True)
        )
    )

    for item in result.results:
        assert item.explanation is not None
        assert len(item.explanation) > 0
```

### 10.3 Reasoning Tests

```python
# tests/test_reasoning.py

async def test_transitive_inference():
    """Test that transitive relations are inferred"""
    # Setup: A PART_OF B, B PART_OF C
    a = create_entity(type="Concept", name="A")
    b = create_entity(type="Concept", name="B")
    c = create_entity(type="Concept", name="C")

    await storage.add_relation(a.id, "PART_OF", b.id)
    await storage.add_relation(b.id, "PART_OF", c.id)

    # Should infer: A PART_OF C
    inferred = await graph_reasoner.infer_relations(a)

    assert any(
        r.target_id == c.id and r.relation_type == "PART_OF" and r.inferred
        for r in inferred
    )
```

---

## Phase 11: Documentation

### 11.1 Core Documentation to Create

1. **VecnaDB Philosophy**
   - Ontology-first design
   - Dual representation principle
   - Graph as truth, vectors as meaning

2. **Ontology Guide**
   - How to define ontologies
   - Entity and relation types
   - Constraints and validation
   - Evolution and migration

3. **Query Guide**
   - Hybrid query construction
   - Ontology filtering
   - Explainability

4. **RAG Guide**
   - Ontology-guided retrieval
   - Hallucination prevention
   - Grounding answers

5. **API Reference**
   - All public endpoints
   - Request/response schemas
   - Examples

6. **Migration Guide**
   - Cognee to VecnaDB
   - Data migration scripts
   - API changes

---

## Implementation Roadmap

### Sprint 1: Foundation (Weeks 1-2)
- [ ] Rename project and modules
- [ ] Refactor DataPoint → KnowledgeEntity
- [ ] Implement EmbeddingRecord model
- [ ] Implement ProvenanceRecord model
- [ ] Update all imports and references

### Sprint 2: Ontology Core (Weeks 3-4)
- [ ] Implement OntologySchema model
- [ ] Implement EntityTypeDefinition and RelationTypeDefinition
- [ ] Implement Constraint system
- [ ] Implement OntologyValidator
- [ ] Create core ontology (YAML)
- [ ] Write ontology validation tests

### Sprint 3: Storage Layer (Weeks 5-6)
- [ ] Design VecnaDBStorageInterface
- [ ] Implement LanceDBKuzuAdapter
- [ ] Implement dual representation enforcement
- [ ] Implement ontology validation in storage layer
- [ ] Write storage integration tests

### Sprint 4: Query System (Weeks 7-8)
- [ ] Implement HybridQuery model
- [ ] Implement SearchResult model
- [ ] Implement HybridQueryExecutor
- [ ] Implement ontology filtering
- [ ] Implement explainability
- [ ] Write query tests

### Sprint 5: Reasoning Layer (Weeks 9-10)
- [ ] Implement GraphReasoner
- [ ] Implement VectorReasoner
- [ ] Implement ReasoningEngine
- [ ] Implement inference rules (transitivity, symmetry)
- [ ] Write reasoning tests

### Sprint 6: RAG Refactor (Weeks 11-12)
- [ ] Implement OntologyGuidedRAG
- [ ] Implement context validation
- [ ] Implement answer grounding
- [ ] Implement OntologyHallucinationPreventer
- [ ] Write RAG tests

### Sprint 7: Versioning & Audit (Weeks 13-14)
- [ ] Implement VersionManager
- [ ] Implement OntologyEvolutionManager
- [ ] Implement AuditLogger
- [ ] Implement metrics collection
- [ ] Write versioning tests

### Sprint 8: API & Documentation (Weeks 15-16)
- [ ] Refactor public API
- [ ] Update API routes
- [ ] Write API documentation
- [ ] Write philosophy documentation
- [ ] Write ontology guide
- [ ] Write migration guide

---

## Success Criteria

### Functional Requirements
✓ All entities MUST have graph node + vector embedding
✓ All entities MUST validate against declared ontology
✓ All queries MUST support ontology constraints
✓ All query results MUST include explanations
✓ All operations MUST be auditable

### Non-Functional Requirements
✓ Query latency < 500ms for 90th percentile
✓ Support 1M+ entities
✓ Zero data loss during ontology evolution
✓ 100% API backward compatibility path

### Quality Requirements
✓ 90%+ test coverage
✓ All public APIs documented
✓ Migration scripts tested
✓ Performance benchmarks established

---

## Guiding Maxim

**Meaning lives in vectors.**
**Truth lives in structure.**
**VecnaDB enforces both.**
