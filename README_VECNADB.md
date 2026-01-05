# VecnaDB

## An Ontology-Native Hybrid Vector-Graph AI Database

**Version:** 0.1.0
**License:** Apache 2.0
**Status:** Active Development (Sprint 1 Complete)

---

## What is VecnaDB?

VecnaDB is not a memory store, chat history tool, or embedding-only database.

It is an **AI-native knowledge database** that unifies:

- **Vector representations** for semantic similarity
- **Graph representations** for structure, logic, and truth
- **Ontologies** as first-class, enforceable schemas

### Guiding Principle

> **Meaning lives in vectors.**
> **Truth lives in structure.**
> **VecnaDB enforces both.**

---

## Core Design Principles

### 1. Dual Representation Principle

Every piece of knowledge MUST:
- Exist as a **typed graph node**
- Have one or more **vector embeddings**

Vectors MUST always reference a graph entity. No "free-floating" embeddings allowed.

### 2. Graph Is Authoritative

Graph structure defines:
- **Validity** - What is allowed
- **Constraints** - What rules apply
- **Truth** - What relationships exist

Vector similarity is **advisory**, never authoritative.

### 3. Ontology-First

All data must conform to a declared ontology:
- Schema-less ingestion is **forbidden**
- All entities and relations are **validated**
- Ontology evolution is **controlled and versioned**

---

## Architecture

### Data Model Hierarchy

```
KnowledgeEntity (Core Model)
├── Identity
│   ├── id: UUID
│   └── type: str
├── Ontology (ENFORCED)
│   ├── ontology_id: UUID
│   ├── ontology_type: str
│   ├── ontology_valid: bool
│   └── validation_errors: List[str]
├── Dual Representation (ENFORCED)
│   ├── graph_node_id: str
│   └── embeddings: List[EmbeddingRecord]  # >= 1 required
├── Versioning
│   ├── version: int
│   ├── supersedes: UUID
│   └── version_metadata: VersionMetadata
└── Auditability
    ├── provenance: ProvenanceRecord
    └── metadata: Dict[str, Any]
```

### Embedding Lifecycle Management

```python
class EmbeddingRecord:
    id: UUID
    entity_id: UUID              # Links to KnowledgeEntity
    embedding_type: EmbeddingType # CONTENT, SUMMARY, ROLE, etc.
    vector: List[float]
    model: str                   # e.g., "text-embedding-3-small"
    model_version: str
    created_at: datetime
    dimensions: int
```

**Supports:**
- Multiple embeddings per entity
- Model version tracking
- Re-embedding without data loss
- Type-specific embeddings (content vs summary)

### Provenance Tracking

```python
class ProvenanceRecord:
    source_document: UUID
    extraction_method: str       # "llm", "manual", "imported"
    extraction_model: str        # LLM used for extraction
    confidence_score: float      # 0.0 - 1.0
    created_by: str
    modified_by: str
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vecnadb.git
cd vecnadb

# Install dependencies
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

---

## Quick Start

### Creating a Knowledge Entity

```python
from vecnadb.infrastructure.engine.models import (
    KnowledgeEntity,
    EmbeddingRecord,
    ProvenanceRecord,
    EmbeddingType
)
from uuid import uuid4

# Create an embedding
embedding = EmbeddingRecord(
    entity_id=uuid4(),  # Will match entity
    embedding_type=EmbeddingType.CONTENT,
    vector=[0.1, 0.2, 0.3, ...],  # Your embedding vector
    model="text-embedding-3-small",
    model_version="1.0",
    dimensions=1536
)

# Create provenance record
provenance = ProvenanceRecord(
    extraction_method="llm",
    extraction_model="gpt-4",
    confidence_score=0.95,
    created_by="data_pipeline"
)

# Create knowledge entity
entity = KnowledgeEntity(
    ontology_id=core_ontology_id,
    ontology_type="Concept",
    embeddings=[embedding],
    provenance=provenance
)

# ✅ Entity is valid - has graph node AND embedding
```

### Validation Examples

```python
# ❌ INVALID: No embeddings
try:
    entity = KnowledgeEntity(
        ontology_id=uuid4(),
        ontology_type="Concept",
        embeddings=[]
    )
except ValueError as e:
    print(e)  # "Dual representation violation: must have at least one embedding"

# ❌ INVALID: Vector dimension mismatch
try:
    embedding = EmbeddingRecord(
        entity_id=uuid4(),
        vector=[0.1, 0.2],  # Only 2 dimensions
        dimensions=3,       # Claims 3 dimensions
        model="test"
    )
except ValueError as e:
    print(e)  # "Vector length 2 does not match declared dimensions 3"
```

---

## Project Status

### Sprint 1: Foundation ✅ COMPLETE

**Completed:**
- ✅ Project renamed from Cognee to VecnaDB
- ✅ Core models created (KnowledgeEntity, EmbeddingRecord, ProvenanceRecord)
- ✅ Dual representation enforcement implemented
- ✅ Vector lifecycle management
- ✅ Provenance tracking
- ✅ Version management
- ✅ 468 files updated with 2,823 import replacements
- ✅ 100% backward compatibility maintained

**Documentation:**
- `VECNADB_REFACTOR_PLAN.md` - Complete refactoring roadmap
- `SPRINT_1_SUMMARY.md` - Detailed Sprint 1 report

### Sprint 2: Ontology Core (NEXT)

**Planned:**
- Ontology schema definition system
- Entity and relation type validation
- Constraint system implementation
- Core ontology (Entity, Concept, Document, Person)
- Ontology validation engine

---

## Key Features

### ✅ Implemented (Sprint 1)

1. **Dual Representation Enforcement**
   - Every entity must have graph node + embedding(s)
   - Validated at model level via Pydantic validators

2. **Vector Lifecycle Management**
   - Multiple embeddings per entity
   - Model version tracking
   - Embedding type classification
   - Re-embedding support

3. **Full Provenance Tracking**
   - Source document tracking
   - Extraction method recording
   - Confidence scores
   - User/system attribution

4. **Version Management**
   - Entity version history
   - Change tracking
   - Supersedes/superseded_by relationships

5. **Type Safety**
   - Pydantic models throughout
   - Field validation
   - Type hints

### 🚧 In Development (Sprint 2+)

1. **Ontology System**
   - Schema definition
   - Entity type validation
   - Relation type validation
   - Constraint enforcement

2. **Hybrid Query System**
   - Vector search + graph traversal
   - Ontology-constrained queries
   - Explainable results

3. **Reasoning Engine**
   - Deterministic (graph-based)
   - Probabilistic (vector-based)
   - Hybrid reasoning

4. **Ontology-Guided RAG**
   - Validated context retrieval
   - Hallucination prevention
   - Grounded answers

---

## Migration from Cognee

VecnaDB maintains 100% backward compatibility with Cognee during the transition period.

### Backward Compatibility

```python
# ✅ Legacy Cognee code still works
from vecnadb.infrastructure.engine.models import DataPoint

data_point = DataPoint(
    # ... existing code
)

# ✅ New VecnaDB code
from vecnadb.infrastructure.engine.models import KnowledgeEntity

entity = KnowledgeEntity(
    # ... new ontology-native code
)
```

### Migration Strategy

1. **Phase 1 (Current):** Both models coexist
2. **Phase 2:** Storage layer updated to use KnowledgeEntity
3. **Phase 3:** API layer migrated
4. **Phase 4:** DataPoint deprecated (with warnings)
5. **Phase 5:** DataPoint removed

---

## Design Philosophy

### What VecnaDB Is NOT

- ❌ Not a chat history system
- ❌ Not an embedding-only database
- ❌ Not a schema-free data store
- ❌ Not optimized for casual memory recall

### What VecnaDB IS

- ✅ An ontology-native knowledge database
- ✅ A truth-preserving graph system
- ✅ A semantic search engine with structure
- ✅ An AI knowledge layer with guarantees

---

## Technical Stack

### Core Dependencies

- **Python:** >= 3.10, < 3.14
- **Pydantic:** >= 2.10.5 (Data validation)
- **LanceDB:** >= 0.24.0 (Vector storage)
- **Kuzu:** == 0.11.3 (Graph storage)
- **LiteLLM:** >= 1.76.0 (LLM integration)

### Optional Backends

- **Neo4j:** Graph database (enterprise)
- **PostgreSQL + pgvector:** Relational + vector
- **ChromaDB:** Vector database
- **AWS Neptune:** Cloud graph database

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test suite
pytest vecnadb/tests/unit/
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy vecnadb/

# Formatting
ruff format .
```

---

## Roadmap

### Sprint 1: Foundation ✅ (Complete)
- Core models and dual representation

### Sprint 2: Ontology Core (2 weeks)
- Ontology schema system
- Validation engine
- Core ontology

### Sprint 3: Storage Layer (2 weeks)
- Unified storage interface
- Dual representation storage
- Migration from Cognee

### Sprint 4: Query System (2 weeks)
- Hybrid queries
- Ontology constraints
- Explainability

### Sprint 5: Reasoning (2 weeks)
- Graph reasoning
- Vector reasoning
- Hybrid orchestration

### Sprint 6: RAG System (2 weeks)
- Ontology-guided RAG
- Hallucination prevention
- Answer grounding

### Sprint 7: Versioning (2 weeks)
- Entity versioning
- Ontology evolution
- Migration tools

### Sprint 8: API & Docs (2 weeks)
- Public API refactor
- Documentation
- Migration guides

---

## Contributing

VecnaDB is under active development. Contributions welcome!

See `CONTRIBUTING.md` for guidelines.

---

## License

Apache License 2.0

---

## Acknowledgments

VecnaDB is a fork of [Cognee](https://github.com/topoteretes/cognee), refactored into an ontology-native architecture.

Original Cognee authors:
- Vasilije Markovic
- Boris Arzentar

VecnaDB maintainers:
- Roberto

---

## Contact

- **Repository:** https://github.com/yourusername/vecnadb
- **Issues:** https://github.com/yourusername/vecnadb/issues

---

## References

- [VecnaDB Refactor Plan](./VECNADB_REFACTOR_PLAN.md)
- [Sprint 1 Summary](./SPRINT_1_SUMMARY.md)
- [Cognee Original Project](https://github.com/topoteretes/cognee)

---

**Status:** Sprint 1 Complete ✅
**Next:** Sprint 2 - Ontology Core Implementation
