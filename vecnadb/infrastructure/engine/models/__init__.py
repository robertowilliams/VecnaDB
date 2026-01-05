"""
VecnaDB Core Models

This module exports the core data models for VecnaDB:
- KnowledgeEntity: New ontology-native entity model (VecnaDB)
- DataPoint: Legacy model (for backward compatibility during transition)
- EmbeddingRecord: Vector embedding lifecycle management
- ProvenanceRecord: Audit trail and provenance tracking
"""

# VecnaDB Models (New)
from .KnowledgeEntity import (
    KnowledgeEntity,
    EmbeddingRecord,
    ProvenanceRecord,
    VersionMetadata,
    EmbeddingType,
    ChangeType,
)

# Legacy Cognee Models (For backward compatibility)
from .DataPoint import DataPoint, MetaData
from .Edge import Edge
from .ExtendableDataPoint import ExtendableDataPoint

__all__ = [
    # VecnaDB Models
    "KnowledgeEntity",
    "EmbeddingRecord",
    "ProvenanceRecord",
    "VersionMetadata",
    "EmbeddingType",
    "ChangeType",
    # Legacy Models
    "DataPoint",
    "MetaData",
    "Edge",
    "ExtendableDataPoint",
]
