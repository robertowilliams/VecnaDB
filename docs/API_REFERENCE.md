# VecnaDB API Reference

Complete API documentation for VecnaDB modules and interfaces.

---

## Core Modules

### Storage

```python
from vecnadb.infrastructure.storage import VecnaDBStorageInterface, LanceDBKuzuAdapter

# Initialize storage
storage = LanceDBKuzuAdapter(
    lancedb_path="./data/vectors",
    kuzu_path="./data/graph"
)

# Register ontology
ontology_id = await storage.register_ontology(ontology)

# Entity operations
entity_id = await storage.add_entity(entity, validate=True)
entity = await storage.get_entity(entity_id)
await storage.update_entity(updated_entity, validate=True)
await storage.delete_entity(entity_id)

# Relation operations
relation_id = await storage.add_relation(source_id, "RELATES_TO", target_id, {})
relations = await storage.get_relations(entity_id, "RELATES_TO", "OUTGOING")

# Search operations
results = await storage.vector_search(query_vector, ["Document"], top_k=10)
subgraph = await storage.graph_search(entity_id, ["IS_A"], max_depth=3)
hybrid_results = await storage.hybrid_search(query_vector, query, ontology_filter)
```

### Query System

```python
from vecnadb.modules.query import HybridQueryBuilder, HybridQueryExecutor

# Build query
query = (
    HybridQueryBuilder()
    .with_query_text("machine learning")
    .with_entity_types(["Document", "Concept"])
    .with_max_results(10)
    .with_graph_depth(2)
    .with_similarity_threshold(0.7)
    .build()
)

# Execute
executor = HybridQueryExecutor(storage, ontology, embedding_service)
results = await executor.execute(query)

for result in results.results:
    print(f"{result.entity.id}: {result.combined_score}")
    print(f"Explanation: {result.explanation}")
```

### Reasoning Engine

```python
from vecnadb.modules.reasoning import ReasoningEngine, ReasoningMode

engine = ReasoningEngine(storage, ontology, embedding_service)

# Hybrid reasoning (facts + suggestions)
result = await engine.reason(
    entity_id=entity.id,
    mode=ReasoningMode.HYBRID,
    max_depth=3,
    top_k_suggestions=10
)

# Facts (confidence = 1.0)
for fact in result.inferred_facts:
    print(f"FACT: {fact.relation_type}")

# Suggestions (confidence < 1.0)
for suggestion in result.entity_suggestions:
    print(f"SUGGESTION: {suggestion.entity.id} ({suggestion.confidence})")
```

### RAG System

```python
from vecnadb.modules.rag import OntologyGuidedRAG, QueryIntent

rag = OntologyGuidedRAG(storage, ontology, llm_service, embedding_service)

# Generate answer
answer = await rag.generate_answer(
    query="What is quantum computing?",
    intent=QueryIntent.DEFINITIONAL,
    require_grounding=True
)

if answer.is_trustworthy(confidence_threshold=0.7):
    print(f"Answer: {answer.answer_text}")
    print(f"Confidence: {answer.confidence}")
    print(f"Sources: {len(answer.context_used)} entities")
```

### Versioning

```python
from vecnadb.modules.versioning import EntityVersioning, ChangeType, ChangeSource

versioning = EntityVersioning(storage)

# Create version
version = await versioning.create_version(
    entity=updated_entity,
    change_type=ChangeType.UPDATE,
    change_source=ChangeSource.USER,
    changed_by="user_123",
    previous_entity=old_entity
)

# Time-travel
past_entity = await versioning.get_entity_at_time(
    entity_id,
    datetime.now() - timedelta(days=7)
)

# Compare versions
diff = await versioning.diff_versions(entity_id, version1=5, version2=10)
```

### Audit Logging

```python
from vecnadb.modules.versioning import AuditLogger, AuditEventType, AuditActor

logger = AuditLogger()
actor = AuditActor(actor_type="user", actor_id="user_123")

# Log operations
await logger.log_entity_operation("update", entity_id, actor, success=True)
await logger.log_rag_operation(query, answer, actor, context, confidence, risk)

# Query logs
events = await logger.query_logs(
    AuditQuery(
        actor_id="user_123",
        start_time=last_week,
        limit=100
    )
)

# Generate report
report = await logger.generate_report(month_start, month_end)
```

See Sprint summaries for detailed examples and use cases.
