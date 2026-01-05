"""
VecnaDB Knowledge Entity Model

This module defines the core KnowledgeEntity model that enforces VecnaDB's
dual representation principle: every knowledge entity MUST exist as both
a typed graph node and have one or more vector embeddings.

Key Principles:
- Ontology-First: All entities must conform to a declared ontology
- Dual Representation: Graph node + Vector embedding(s)
- Full Auditability: Complete provenance tracking
- Versioning: Full version history support
"""

from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator


# Enums for type safety
class EmbeddingType(str, Enum):
    """Types of embeddings that can be associated with an entity"""
    CONTENT = "content"  # Embedding of main content
    SUMMARY = "summary"  # Embedding of summarized content
    ROLE = "role"  # Role-based embedding for specific use cases
    TITLE = "title"  # Embedding of title/name only
    METADATA = "metadata"  # Embedding of metadata fields
    CUSTOM = "custom"  # Custom embedding type


class ChangeType(str, Enum):
    """Types of changes for version tracking"""
    CREATED = "created"
    UPDATED = "updated"
    MERGED = "merged"
    SPLIT = "split"
    DELETED = "deleted"


# Supporting Models
class EmbeddingRecord(BaseModel):
    """
    Vector embedding record with lifecycle management.

    Each embedding is versioned and tracks the model that created it,
    enabling re-embedding and model migration without data loss.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    entity_id: UUID  # MUST reference a KnowledgeEntity
    embedding_type: EmbeddingType = EmbeddingType.CONTENT
    vector: List[float]
    model: str  # e.g., "text-embedding-3-small", "sentence-transformers/all-MiniLM-L6-v2"
    model_version: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dimensions: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('vector')
    @classmethod
    def validate_vector_dimensions(cls, v, info):
        """Ensure vector has correct dimensions"""
        if 'dimensions' in info.data and len(v) != info.data['dimensions']:
            raise ValueError(
                f"Vector length {len(v)} does not match declared dimensions {info.data['dimensions']}"
            )
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()


class ProvenanceRecord(BaseModel):
    """
    Provenance tracking for full auditability.

    Records the origin and creation history of each knowledge entity,
    enabling complete reconstruction of how knowledge was acquired.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_document: Optional[UUID] = None  # Source document ID if applicable
    extraction_method: str  # "llm", "manual", "imported", "inferred"
    extraction_model: Optional[str] = None  # LLM model used if applicable
    confidence_score: Optional[float] = None  # Confidence in extraction (0.0-1.0)
    created_by: Optional[str] = None  # User or system that created entity
    modified_by: Optional[str] = None  # Last user/system to modify
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()


class VersionMetadata(BaseModel):
    """
    Metadata for entity versioning.

    Tracks what changed, why, and when for full version history.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    change_type: ChangeType
    changed_properties: List[str] = Field(default_factory=list)
    change_reason: str = "No reason provided"
    changed_by: str = "system"
    changed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()


class KnowledgeEntity(BaseModel):
    """
    Core VecnaDB Knowledge Entity Model.

    Enforces the dual representation principle:
    - Every entity MUST be a typed graph node
    - Every entity MUST have at least one vector embedding
    - Every entity MUST conform to a declared ontology

    This model replaces Cognee's DataPoint with stronger guarantees
    around ontology compliance and structural integrity.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ========== Identity ==========
    id: UUID = Field(default_factory=uuid4)
    type: str  # Python class name (for polymorphism)

    # ========== Ontology Enforcement ==========
    ontology_id: UUID  # Reference to ontology version
    ontology_type: str  # Entity type from ontology (e.g., "Concept", "Person")
    ontology_valid: bool = False  # Validation status
    validation_errors: Optional[List[str]] = None  # Validation error messages

    # ========== Temporal Tracking ==========
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    supersedes: Optional[UUID] = None  # Previous version ID
    superseded_by: Optional[UUID] = None  # Next version ID (set when superseded)
    version_metadata: Optional[VersionMetadata] = None

    # ========== Graph Properties ==========
    graph_node_id: str  # Unique identifier in graph database
    topological_rank: Optional[int] = 0  # Graph topology ranking

    # ========== Vector Properties ==========
    # CRITICAL: Must have at least one embedding (enforced in validator)
    embeddings: List[EmbeddingRecord] = Field(default_factory=list)

    # ========== Metadata ==========
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[ProvenanceRecord] = None

    def __init__(self, **data):
        """Initialize and set type to class name"""
        super().__init__(**data)
        # Set graph_node_id to str(id) if not provided
        if not data.get('graph_node_id'):
            object.__setattr__(self, 'graph_node_id', str(self.id))
        # Set type to class name
        object.__setattr__(self, 'type', self.__class__.__name__)

    @field_validator('embeddings')
    @classmethod
    def validate_embeddings_exist(cls, v):
        """
        Enforce dual representation: every entity MUST have at least one embedding.

        This is a core VecnaDB principle - no free-floating entities without
        vector representations are allowed.
        """
        if not v or len(v) == 0:
            raise ValueError(
                "Dual representation violation: KnowledgeEntity must have at least one embedding. "
                "Vectors are required for semantic operations."
            )
        return v

    def get_embedding(self, embedding_type: EmbeddingType = EmbeddingType.CONTENT) -> Optional[EmbeddingRecord]:
        """
        Retrieve a specific embedding by type.

        Args:
            embedding_type: Type of embedding to retrieve

        Returns:
            EmbeddingRecord if found, None otherwise
        """
        for embedding in self.embeddings:
            if embedding.embedding_type == embedding_type:
                return embedding
        return None

    def get_all_embedding_types(self) -> List[EmbeddingType]:
        """Get all embedding types present in this entity"""
        return [e.embedding_type for e in self.embeddings]

    def add_embedding(self, embedding: EmbeddingRecord) -> None:
        """
        Add a new embedding to this entity.

        Args:
            embedding: EmbeddingRecord to add

        Raises:
            ValueError: If embedding.entity_id doesn't match this entity's ID
        """
        if embedding.entity_id != self.id:
            raise ValueError(
                f"Embedding entity_id {embedding.entity_id} does not match "
                f"KnowledgeEntity id {self.id}"
            )
        self.embeddings.append(embedding)
        self.update_version(reason="Added embedding")

    def update_version(self, reason: str = "Manual update", changed_by: str = "system") -> None:
        """
        Increment version and update metadata.

        Args:
            reason: Reason for version update
            changed_by: User or system making the change
        """
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)
        self.version_metadata = VersionMetadata(
            change_type=ChangeType.UPDATED,
            change_reason=reason,
            changed_by=changed_by,
            changed_at=self.updated_at
        )

    def get_embeddable_properties(self) -> List[Any]:
        """
        Get properties that should be embedded.

        For backward compatibility with Cognee's DataPoint interface.
        Extracts properties based on metadata index_fields if present.
        """
        if "index_fields" in self.metadata and self.metadata["index_fields"]:
            return [
                getattr(self, field, None)
                for field in self.metadata["index_fields"]
            ]
        return []

    def get_embeddable_property_names(self) -> List[str]:
        """
        Get names of embeddable properties.

        For backward compatibility with Cognee's DataPoint interface.
        """
        if "index_fields" in self.metadata:
            return self.metadata["index_fields"]
        return []

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump(**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntity":
        """Create instance from dictionary"""
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeEntity":
        """Deserialize from JSON string"""
        return cls.model_validate_json(json_str)

    def __repr__(self) -> str:
        return (
            f"KnowledgeEntity(id={self.id}, type={self.ontology_type}, "
            f"embeddings={len(self.embeddings)}, valid={self.ontology_valid})"
        )
