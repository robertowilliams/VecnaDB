# VecnaDB

## An Ontology-Native Hybrid Vector-Graph AI Database

**Version:** 0.1.0
**License:** Apache 2.0
**Status:** Production Ready - All 8 Sprints Complete ✅

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

## Key Features

### ✅ Implemented - All 8 Sprints Complete

#### 1. Foundation (Sprint 1)
- **Dual Representation Enforcement**: Every entity has graph node + embedding(s)
- **Vector Lifecycle Management**: Multiple embeddings, model tracking, re-embedding
- **Full Provenance Tracking**: Source documents, extraction methods, confidence scores
- **Version Management**: Entity version history, change tracking
- **Type Safety**: Pydantic models throughout

#### 2. Ontology System (Sprint 2)
- **Schema Definition**: Comprehensive ontology modeling with inheritance
- **Entity & Relation Validation**: Type checking, constraint enforcement
- **Core Ontology**: 8 built-in entity types, 12 relation types
- **Constraint System**: REGEX, RANGE, ENUM, REQUIRED, UNIQUE, etc.

#### 3. Storage Layer (Sprint 3)
- **Unified Storage Interface**: Single API for all operations (40+ methods)
- **LanceDB + Kuzu Backend**: Columnar vectors + embedded graph database
- **Atomic Dual Writes**: Guaranteed consistency across vector and graph stores
- **Ontology Validation at Storage**: All writes validated before persistence

#### 4. Hybrid Query System (Sprint 4)
- **Vector + Graph Search**: Combined semantic similarity and structural traversal
- **Ontology-Constrained Queries**: Filter by types, relations, constraints
- **7-Step Query Pipeline**: Embed → Vector Search → Filter → Expand → Rank → Explain → Return
- **Mandatory Explainability**: Every result includes reasoning trace

#### 5. Reasoning Engine (Sprint 5)
- **Graph Reasoning**: Deterministic inference (transitive, symmetric) with confidence = 1.0
- **Vector Reasoning**: Probabilistic suggestions (similarity, analogy) with confidence < 1.0
- **Hybrid Orchestration**: Four reasoning modes (GRAPH_ONLY, VECTOR_ONLY, HYBRID, SEQUENTIAL)
- **Explainable Inferences**: All reasoning includes step-by-step justifications

#### 6. Ontology-Guided RAG (Sprint 6)
- **RAG with Hallucination Prevention**: Claim extraction, entity grounding, relation verification
- **Context Validation**: Multi-level quality checks (STRICT, MODERATE, LENIENT)
- **Answer Grounding**: Complete provenance with citations (INLINE, SUPERSCRIPT, etc.)
- **Confidence Scoring**: Risk assessment for all generated answers

#### 7. Versioning & Audit Trail (Sprint 7)
- **Entity Versioning**: Immutable version history with time-travel queries
- **Ontology Evolution**: Semantic versioning (MAJOR.MINOR.PATCH) for schemas
- **Migration Tools**: Data migration execution with rollback support
- **Complete Audit Logging**: 15+ event types, compliance reporting, full traceability

#### 8. Documentation (Sprint 8)
- **Comprehensive README**: Quick start, architecture, use cases
- **Architecture Guide**: Deep technical documentation (800+ lines)
- **API Reference**: Code examples for all major modules
- **Migration Guide**: Operational guide for schema migrations

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

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  (RAG, Reasoning, Query, Ingestion Pipelines)           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   VecnaDB Core Modules                   │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │ Ontology │  Query   │Reasoning │  RAG + Halluc.   │  │
│  │ Validator│ Executor │  Engine  │    Prevention    │  │
│  └──────────┴──────────┴──────────┴──────────────────┘  │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │ Version  │  Audit   │Migration │   Storage API    │  │
│  │  Track   │  Logger  │  Tools   │  (40+ methods)   │  │
│  └──────────┴──────────┴──────────┴──────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Dual Storage Backends                       │
│  ┌─────────────────────┬─────────────────────────────┐  │
│  │   LanceDB           │         Kuzu                │  │
│  │ (Vector Embeddings) │    (Knowledge Graph)        │  │
│  │                     │                             │  │
│  │ • Columnar storage  │ • Typed nodes & relations   │  │
│  │ • ANN search        │ • ACID transactions         │  │
│  │ • Multi-embedding   │ • Cypher-like queries       │  │
│  └─────────────────────┴─────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
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

### Requirements

- **Python:** >= 3.10, < 3.14
- **Pydantic:** >= 2.10.5 (Data validation)
- **LanceDB:** >= 0.24.0 (Vector storage)
- **Kuzu:** == 0.11.3 (Graph storage)
- **LiteLLM:** >= 1.76.0 (LLM integration)

---

## Quick Start

### 1. Basic Entity Creation

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
    entity_id=uuid4(),
    embedding_type=EmbeddingType.CONTENT,
    vector=[0.1, 0.2, 0.3, ...],  # Your 1536-dim vector
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
```

### 2. Hybrid Query

```python
from vecnadb.modules.query.models import HybridQuery, HybridQueryBuilder
from vecnadb.modules.query.executor import HybridQueryExecutor

# Build a query
query = (
    HybridQueryBuilder()
    .with_text("What is machine learning?")
    .with_entity_types(["Concept", "Document"])
    .with_max_depth(2)
    .with_max_results(10)
    .build()
)

# Execute hybrid search
executor = HybridQueryExecutor(storage=storage)
results = await executor.execute(query)

# Results include:
# - Matched entities (vector + graph)
# - Explanations for each result
# - Confidence scores
# - Provenance information
```

### 3. Reasoning

```python
from vecnadb.modules.reasoning import ReasoningEngine, ReasoningMode

engine = ReasoningEngine(storage=storage)

# Hybrid reasoning (deterministic + probabilistic)
result = await engine.reason(
    entity_id=concept_id,
    mode=ReasoningMode.HYBRID,
    max_depth=3
)

# result.facts = graph-based inferences (confidence=1.0)
# result.suggestions = vector-based suggestions (confidence<1.0)
# result.explanations = step-by-step reasoning trace
```

### 4. RAG with Hallucination Prevention

```python
from vecnadb.modules.rag import OntologyGuidedRAG, RAGIntent

rag = OntologyGuidedRAG(storage=storage)

# Generate grounded answer
answer = await rag.generate_answer(
    query="Explain the relationship between AI and machine learning",
    intent=RAGIntent.EXPLAIN,
    require_grounding=True
)

# answer.text = LLM-generated response
# answer.grounding.sources = knowledge graph entities used
# answer.grounding.citations = provenance links
# answer.hallucination_report.risk_score = hallucination risk (0.0-1.0)
# answer.confidence = overall answer confidence
```

### 5. Time-Travel Queries

```python
from vecnadb.modules.versioning import EntityVersioning
from datetime import datetime

versioning = EntityVersioning(storage=storage)

# Get entity state at specific time
past_entity = await versioning.get_entity_at_time(
    entity_id=entity_id,
    timestamp=datetime(2025, 1, 1)
)

# Compare versions
diff = await versioning.diff_versions(
    entity_id=entity_id,
    from_version=1,
    to_version=3
)

# diff.changes = property-level differences
```

---

## Use Cases

### 1. Enterprise Knowledge Management
- Build company-wide knowledge graphs with ontology enforcement
- Track document provenance and extraction confidence
- Version control for knowledge evolution
- Audit trail for compliance (GDPR, SOC2)

### 2. Scientific Research
- Model domain-specific ontologies (biomedical, chemistry, physics)
- Link publications, concepts, and experiments
- Reason over research graphs with explainability
- Prevent hallucinations in literature review generation

### 3. Financial Intelligence
- Model entities (companies, people, products) with strict schemas
- Track relationship changes over time
- Audit all data transformations for regulatory compliance
- Generate investment insights grounded in source documents

### 4. Legal & Compliance
- Build case law and regulatory knowledge graphs
- Citation-backed legal reasoning
- Track precedent evolution over time
- Generate compliant summaries with full provenance

### 5. AI Agent Memory
- Persistent, structured memory for autonomous agents
- Reason over past interactions and learned knowledge
- Ground agent responses in verified facts
- Audit agent decision-making process

---

## Project Statistics

- **~12,000 lines** of production code
- **~5,000 lines** of documentation
- **40+ storage methods** in unified interface
- **8 entity types**, **12 relation types** in core ontology
- **15+ audit event types**
- **4 reasoning modes** (GRAPH_ONLY, VECTOR_ONLY, HYBRID, SEQUENTIAL)
- **3 validation levels** (STRICT, MODERATE, LENIENT)
- **5 citation styles** for answer grounding

---

## Roadmap

### ✅ Phase 1: Core Platform (Complete)
- Sprint 1: Foundation
- Sprint 2: Ontology Core
- Sprint 3: Storage Layer
- Sprint 4: Hybrid Query
- Sprint 5: Reasoning Engine
- Sprint 6: RAG System
- Sprint 7: Versioning & Audit
- Sprint 8: Documentation

### 🚀 Phase 2: API & Integration (Future)
- REST API with FastAPI
- GraphQL endpoint
- WebSocket support for real-time updates
- Python SDK refinement
- Client libraries (JavaScript, Go)

### 🔮 Phase 3: Advanced Features (Future)
- Multi-tenancy support
- Distributed deployment
- Advanced query optimization
- Real-time graph updates
- Federated learning support
- Custom reasoning plugins

---

## Design Philosophy

### What VecnaDB Is NOT

- ❌ Not a chat history system
- ❌ Not an embedding-only database
- ❌ Not a schema-free data store
- ❌ Not optimized for casual memory recall
- ❌ Not a replacement for traditional SQL databases

### What VecnaDB IS

- ✅ An ontology-native knowledge database
- ✅ A truth-preserving graph system with semantic search
- ✅ A semantic search engine with structural guarantees
- ✅ An AI knowledge layer with explainability
- ✅ A foundation for trustworthy AI applications

---

## Documentation

- **[Architecture Guide](./docs/ARCHITECTURE.md)** - Deep technical documentation
- **[API Reference](./docs/API_REFERENCE.md)** - Code examples for all modules
- **[Migration Guide](./docs/MIGRATION_GUIDE.md)** - Ontology migration operations
- **Sprint Summaries**: [Sprint 1](./SPRINT_1_SUMMARY.md) through [Sprint 8](./SPRINT_8_SUMMARY.md)

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

# Run with coverage
pytest --cov=vecnadb
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

## Migration from Cognee

VecnaDB maintains backward compatibility with Cognee's legacy `DataPoint` model during the transition period.

```python
# ✅ Legacy Cognee code still works
from vecnadb.infrastructure.engine.models import DataPoint

data_point = DataPoint(
    # ... existing code
)

# ✅ New VecnaDB code (recommended)
from vecnadb.infrastructure.engine.models import KnowledgeEntity

entity = KnowledgeEntity(
    # ... new ontology-native code
)
```

See [Migration Guide](./docs/MIGRATION_GUIDE.md) for detailed migration instructions.

---

## Contributing

VecnaDB is open-source and welcomes contributions!

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for guidelines.

---

## License

Apache License 2.0

See [`LICENSE`](./LICENSE) for full text.

---

## Acknowledgments

VecnaDB is a fork of [Cognee](https://github.com/topoteretes/cognee), refactored into an ontology-native architecture.

**Original Cognee Authors:**
- Vasilije Markovic
- Boris Arzentar

**VecnaDB Maintainer:**
- Roberto

Special thanks to the Cognee community for creating the foundation that made VecnaDB possible.

---

## Contact

- **Repository:** https://github.com/yourusername/vecnadb
- **Issues:** https://github.com/yourusername/vecnadb/issues
- **Discussions:** https://github.com/yourusername/vecnadb/discussions

---

## References

- [Cognee Original Project](https://github.com/topoteretes/cognee)
- [VecnaDB Refactor Plan](./VECNADB_REFACTOR_PLAN.md)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Kuzu Documentation](https://kuzudb.com/)

---

**Status:** Production Ready ✅
**All 8 Sprints Complete**
**~12,000 lines of code | ~5,000 lines of documentation**
