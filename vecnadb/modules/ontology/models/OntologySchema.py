"""
VecnaDB Ontology Schema Models

This module defines the ontology schema system that enables VecnaDB's
ontology-first design. All knowledge entities and relations must conform
to declared ontology schemas.

Key Principles:
- Ontology defines what is valid
- Entities and relations must conform to their types
- Constraints are enforced at validation time
- Ontologies are versioned and can evolve
"""

from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


# Enums for ontology system
class PropertyType(str, Enum):
    """Data types for entity and relation properties"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    UUID = "uuid"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    ANY = "any"


class ConstraintType(str, Enum):
    """Types of constraints that can be applied"""
    REGEX = "regex"              # String pattern matching
    RANGE = "range"              # Numeric range (min, max)
    ENUM = "enum"                # Must be one of specified values
    UNIQUE = "unique"            # Value must be unique across entities
    REFERENCE = "reference"      # Must reference another entity
    CUSTOM = "custom"            # Custom validation function
    LENGTH = "length"            # String/list length constraints
    REQUIRED_IF = "required_if"  # Required if condition met


class Cardinality(str, Enum):
    """Cardinality for relationships"""
    ONE = "one"          # Exactly one
    ZERO_OR_ONE = "0..1"  # Optional, at most one
    ZERO_OR_MORE = "0..*" # Any number
    ONE_OR_MORE = "1..*"  # At least one


# Supporting Models
class Constraint(BaseModel):
    """
    A constraint that can be applied to properties, entities, or relations.

    Examples:
    - Regex: {"type": "regex", "parameters": {"pattern": "^[A-Z].*"}}
    - Range: {"type": "range", "parameters": {"min": 0, "max": 100}}
    - Enum: {"type": "enum", "parameters": {"values": ["ACTIVE", "INACTIVE"]}}
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: ConstraintType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    error_message: str = "Constraint validation failed"

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a value against this constraint.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if self.type == ConstraintType.REGEX:
                import re
                pattern = self.parameters.get("pattern", "")
                if not isinstance(value, str):
                    return False, f"Value must be string for regex constraint"
                if not re.match(pattern, value):
                    return False, self.error_message
                return True, None

            elif self.type == ConstraintType.RANGE:
                min_val = self.parameters.get("min")
                max_val = self.parameters.get("max")
                if min_val is not None and value < min_val:
                    return False, f"Value {value} is less than minimum {min_val}"
                if max_val is not None and value > max_val:
                    return False, f"Value {value} exceeds maximum {max_val}"
                return True, None

            elif self.type == ConstraintType.ENUM:
                allowed_values = self.parameters.get("values", [])
                if value not in allowed_values:
                    return False, f"Value {value} not in allowed values: {allowed_values}"
                return True, None

            elif self.type == ConstraintType.LENGTH:
                min_len = self.parameters.get("min")
                max_len = self.parameters.get("max")
                length = len(value) if hasattr(value, '__len__') else None
                if length is None:
                    return False, "Value has no length"
                if min_len is not None and length < min_len:
                    return False, f"Length {length} is less than minimum {min_len}"
                if max_len is not None and length > max_len:
                    return False, f"Length {length} exceeds maximum {max_len}"
                return True, None

            elif self.type == ConstraintType.UNIQUE:
                # Note: Uniqueness validation requires access to storage
                # This will be implemented in the validator class
                return True, None

            elif self.type == ConstraintType.REFERENCE:
                # Note: Reference validation requires access to storage
                # This will be implemented in the validator class
                return True, None

            else:
                # Unknown constraint type or custom constraint
                return True, None

        except Exception as e:
            return False, f"Constraint validation error: {str(e)}"


class PropertyDefinition(BaseModel):
    """
    Definition of a property that can exist on entities or relations.

    Properties define the data fields that entities can have,
    with type constraints and validation rules.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    type: PropertyType
    required: bool = False
    default: Optional[Any] = None
    constraints: List[Constraint] = Field(default_factory=list)
    indexed: bool = False  # Whether this property should be indexed
    embeddable: bool = False  # Whether to include in vector embeddings
    description: str = ""

    def validate_value(self, value: Any) -> tuple[bool, List[str]]:
        """
        Validate a value against this property definition.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Type validation
        if not self._validate_type(value):
            errors.append(
                f"Property '{self.name}': Expected type {self.type.value}, "
                f"got {type(value).__name__}"
            )

        # Constraint validation
        for constraint in self.constraints:
            is_valid, error = constraint.validate(value)
            if not is_valid:
                errors.append(f"Property '{self.name}': {error}")

        return len(errors) == 0, errors

    def _validate_type(self, value: Any) -> bool:
        """Validate that value matches the expected type"""
        if self.type == PropertyType.STRING:
            return isinstance(value, str)
        elif self.type == PropertyType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        elif self.type == PropertyType.FLOAT:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif self.type == PropertyType.BOOLEAN:
            return isinstance(value, bool)
        elif self.type == PropertyType.UUID:
            return isinstance(value, (UUID, str))
        elif self.type == PropertyType.DATETIME:
            return isinstance(value, datetime)
        elif self.type == PropertyType.LIST:
            return isinstance(value, list)
        elif self.type == PropertyType.DICT:
            return isinstance(value, dict)
        elif self.type == PropertyType.ANY:
            return True
        return False


class EmbeddingRequirements(BaseModel):
    """
    Requirements for embeddings on this entity type.

    Defines which embedding types are required and which properties
    should be included in embeddings.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    required_types: List[str] = Field(default_factory=lambda: ["content"])
    optional_types: List[str] = Field(default_factory=list)
    embeddable_properties: List[str] = Field(default_factory=list)
    min_embeddings: int = 1  # Minimum number of embeddings required


class EntityTypeDefinition(BaseModel):
    """
    Definition of an entity type in the ontology.

    Entity types define the structure and constraints for knowledge entities,
    including what properties they can have and what rules they must follow.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    properties: Dict[str, PropertyDefinition] = Field(default_factory=dict)
    required_properties: List[str] = Field(default_factory=list)
    inherits_from: Optional[List[str]] = None
    constraints: List[Constraint] = Field(default_factory=list)
    embedding_requirements: EmbeddingRequirements = Field(
        default_factory=EmbeddingRequirements
    )
    abstract: bool = False  # Abstract types cannot be instantiated

    def get_all_properties(
        self,
        ontology: "OntologySchema"
    ) -> Dict[str, PropertyDefinition]:
        """
        Get all properties including inherited ones.

        Args:
            ontology: The ontology schema containing parent types

        Returns:
            Dictionary of all properties (own + inherited)
        """
        all_props = {}

        # Add inherited properties first
        if self.inherits_from:
            for parent_name in self.inherits_from:
                if parent_name in ontology.entity_types:
                    parent = ontology.entity_types[parent_name]
                    parent_props = parent.get_all_properties(ontology)
                    all_props.update(parent_props)

        # Override with own properties
        all_props.update(self.properties)

        return all_props

    def get_all_required_properties(
        self,
        ontology: "OntologySchema"
    ) -> List[str]:
        """Get all required properties including inherited ones"""
        required = set(self.required_properties)

        if self.inherits_from:
            for parent_name in self.inherits_from:
                if parent_name in ontology.entity_types:
                    parent = ontology.entity_types[parent_name]
                    required.update(parent.get_all_required_properties(ontology))

        return list(required)


class RelationTypeDefinition(BaseModel):
    """
    Definition of a relation type in the ontology.

    Relation types define what relationships can exist between entities,
    with constraints on source/target types and cardinality rules.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""

    # Type constraints
    allowed_source_types: List[str] = Field(default_factory=list)
    allowed_target_types: List[str] = Field(default_factory=list)

    # Cardinality constraints
    source_cardinality: Cardinality = Cardinality.ZERO_OR_MORE
    target_cardinality: Cardinality = Cardinality.ZERO_OR_MORE

    # Directionality
    is_directed: bool = True
    symmetric: bool = False  # If A→B, then B→A
    transitive: bool = False  # If A→B and B→C, then A→C
    inverse_of: Optional[str] = None  # Name of inverse relation

    # Properties on the relation itself
    properties: Dict[str, PropertyDefinition] = Field(default_factory=dict)
    required_properties: List[str] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)

    def allows_source_type(self, entity_type: str, ontology: "OntologySchema") -> bool:
        """
        Check if this entity type is allowed as a source.

        Considers inheritance hierarchy.
        """
        if not self.allowed_source_types:
            return True  # No restriction

        # Direct match
        if entity_type in self.allowed_source_types:
            return True

        # Check inheritance
        if entity_type in ontology.entity_types:
            entity_def = ontology.entity_types[entity_type]
            if entity_def.inherits_from:
                for parent in entity_def.inherits_from:
                    if self.allows_source_type(parent, ontology):
                        return True

        return False

    def allows_target_type(self, entity_type: str, ontology: "OntologySchema") -> bool:
        """
        Check if this entity type is allowed as a target.

        Considers inheritance hierarchy.
        """
        if not self.allowed_target_types:
            return True  # No restriction

        # Direct match
        if entity_type in self.allowed_target_types:
            return True

        # Check inheritance
        if entity_type in ontology.entity_types:
            entity_def = ontology.entity_types[entity_type]
            if entity_def.inherits_from:
                for parent in entity_def.inherits_from:
                    if self.allows_target_type(parent, ontology):
                        return True

        return False


class OntologySchema(BaseModel):
    """
    Complete ontology schema definition.

    An ontology defines:
    - Entity types (what kinds of knowledge entities exist)
    - Relation types (what relationships are allowed)
    - Inheritance rules (type hierarchy)
    - Constraints (validation rules)

    Ontologies are versioned and can evolve over time.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identity
    id: UUID = Field(default_factory=uuid4)
    name: str
    version: str
    description: str = ""

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Versioning
    supersedes: Optional[UUID] = None  # Previous ontology version
    superseded_by: Optional[UUID] = None  # Next version

    # Type definitions
    entity_types: Dict[str, EntityTypeDefinition] = Field(default_factory=dict)
    relation_types: Dict[str, RelationTypeDefinition] = Field(default_factory=dict)

    # Inheritance graph (child → parents)
    # This is derived from entity_types but cached for performance
    inheritance_graph: Dict[str, List[str]] = Field(default_factory=dict)

    # Global constraints that apply to all entities
    global_constraints: List[Constraint] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        # Build inheritance graph from entity types
        self._build_inheritance_graph()

    def _build_inheritance_graph(self):
        """Build inheritance graph from entity type definitions"""
        self.inheritance_graph = {}
        for type_name, type_def in self.entity_types.items():
            if type_def.inherits_from:
                self.inheritance_graph[type_name] = type_def.inherits_from

    def get_entity_type(self, type_name: str) -> Optional[EntityTypeDefinition]:
        """Get entity type definition by name"""
        return self.entity_types.get(type_name)

    def get_relation_type(self, type_name: str) -> Optional[RelationTypeDefinition]:
        """Get relation type definition by name"""
        return self.relation_types.get(type_name)

    def add_entity_type(self, type_def: EntityTypeDefinition):
        """Add a new entity type to the ontology"""
        self.entity_types[type_def.name] = type_def
        if type_def.inherits_from:
            self.inheritance_graph[type_def.name] = type_def.inherits_from
        self.updated_at = datetime.now(timezone.utc)

    def add_relation_type(self, type_def: RelationTypeDefinition):
        """Add a new relation type to the ontology"""
        self.relation_types[type_def.name] = type_def
        self.updated_at = datetime.now(timezone.utc)

    def get_type_hierarchy(self, type_name: str) -> List[str]:
        """
        Get complete type hierarchy for an entity type.

        Returns list from most specific to most general.
        Example: ["Person", "Agent", "Entity"]
        """
        hierarchy = [type_name]
        current = type_name

        while current in self.inheritance_graph:
            parents = self.inheritance_graph[current]
            if not parents:
                break
            # Add first parent to hierarchy
            parent = parents[0]
            hierarchy.append(parent)
            current = parent

        return hierarchy

    def is_subtype_of(self, child_type: str, parent_type: str) -> bool:
        """Check if child_type is a subtype of parent_type"""
        if child_type == parent_type:
            return True

        hierarchy = self.get_type_hierarchy(child_type)
        return parent_type in hierarchy

    def validate_entity_type_exists(self, type_name: str) -> tuple[bool, Optional[str]]:
        """Validate that entity type exists in ontology"""
        if type_name not in self.entity_types:
            return False, f"Unknown entity type: {type_name}"
        return True, None

    def validate_relation_type_exists(self, type_name: str) -> tuple[bool, Optional[str]]:
        """Validate that relation type exists in ontology"""
        if type_name not in self.relation_types:
            return False, f"Unknown relation type: {type_name}"
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologySchema":
        """Create from dictionary representation"""
        return cls.model_validate(data)

    def __repr__(self) -> str:
        return (
            f"OntologySchema(name='{self.name}', version='{self.version}', "
            f"entities={len(self.entity_types)}, relations={len(self.relation_types)})"
        )
