"""
VecnaDB Ontology Module

This module provides ontology management functionality for VecnaDB's
ontology-first design.

Key exports:
- OntologySchema: Core ontology schema model
- OntologyValidator: Validation engine
- OntologyLoader: Load ontologies from YAML/JSON
- load_core_ontology: Convenience function for core ontology
"""

from vecnadb.modules.ontology.models.OntologySchema import (
    OntologySchema,
    EntityTypeDefinition,
    RelationTypeDefinition,
    PropertyDefinition,
    Constraint,
    EmbeddingRequirements,
    PropertyType,
    ConstraintType,
    Cardinality,
)

from vecnadb.modules.ontology.validation.OntologyValidator import (
    OntologyValidator,
    ValidationResult,
)

from vecnadb.modules.ontology.loaders.OntologyLoader import (
    OntologyLoader,
    load_core_ontology,
)

__all__ = [
    # Schema models
    "OntologySchema",
    "EntityTypeDefinition",
    "RelationTypeDefinition",
    "PropertyDefinition",
    "Constraint",
    "EmbeddingRequirements",
    # Enums
    "PropertyType",
    "ConstraintType",
    "Cardinality",
    # Validation
    "OntologyValidator",
    "ValidationResult",
    # Loading
    "OntologyLoader",
    "load_core_ontology",
]
