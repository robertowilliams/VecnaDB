"""
VecnaDB Ontology Validator

This module implements ontology validation logic that enforces VecnaDB's
ontology-first principle. All entities and relations must pass validation
before being stored.

Key Responsibilities:
- Validate entities against entity type definitions
- Validate relations against relation type definitions
- Check constraints and rules
- Provide detailed error messages
"""

from typing import List, Optional, Dict, Any
from uuid import UUID

from vecnadb.modules.ontology.models.OntologySchema import (
    OntologySchema,
    EntityTypeDefinition,
    RelationTypeDefinition,
    Constraint,
    ConstraintType,
)


class ValidationResult(BaseModel):
    """Result of ontology validation"""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult"):
        """Merge another validation result into this one"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False


class OntologyValidator:
    """
    Validates knowledge entities and relations against ontology schemas.

    This is the enforcement mechanism for VecnaDB's ontology-first design.
    No entity or relation can be stored without passing validation.
    """

    def __init__(self, ontology: OntologySchema, storage=None):
        """
        Initialize validator with an ontology schema.

        Args:
            ontology: The ontology schema to validate against
            storage: Optional storage interface for uniqueness/reference checks
        """
        self.ontology = ontology
        self.storage = storage

    async def validate_entity(
        self,
        entity: "KnowledgeEntity"
    ) -> ValidationResult:
        """
        Validate a knowledge entity against the ontology.

        Checks:
        1. Entity type exists in ontology
        2. Entity type is not abstract
        3. Required properties are present
        4. Property types match
        5. Property constraints are satisfied
        6. Entity-level constraints are satisfied
        7. Embedding requirements are met

        Args:
            entity: The KnowledgeEntity to validate

        Returns:
            ValidationResult with validation status and error messages
        """
        result = ValidationResult(valid=True)

        # Check 1: Entity type exists
        if entity.ontology_type not in self.ontology.entity_types:
            result.add_error(
                f"Unknown entity type: '{entity.ontology_type}'. "
                f"Available types: {list(self.ontology.entity_types.keys())}"
            )
            return result

        type_def = self.ontology.entity_types[entity.ontology_type]

        # Check 2: Type is not abstract
        if type_def.abstract:
            result.add_error(
                f"Cannot instantiate abstract entity type: '{entity.ontology_type}'"
            )

        # Check 3: Required properties
        required_props = type_def.get_all_required_properties(self.ontology)
        for prop_name in required_props:
            if not hasattr(entity, prop_name) or getattr(entity, prop_name) is None:
                result.add_error(
                    f"Missing required property: '{prop_name}' "
                    f"for type '{entity.ontology_type}'"
                )

        # Check 4 & 5: Property validation
        all_props = type_def.get_all_properties(self.ontology)
        for prop_name, prop_def in all_props.items():
            if hasattr(entity, prop_name):
                value = getattr(entity, prop_name)
                if value is not None:  # Only validate if value is present
                    is_valid, errors = prop_def.validate_value(value)
                    if not is_valid:
                        for error in errors:
                            result.add_error(error)

        # Check 6: Entity-level constraints
        for constraint in type_def.constraints:
            is_valid, error = constraint.validate(entity)
            if not is_valid:
                result.add_error(
                    f"Entity constraint failed for '{entity.ontology_type}': {error}"
                )

        # Check 7: Embedding requirements
        emb_req = type_def.embedding_requirements
        if len(entity.embeddings) < emb_req.min_embeddings:
            result.add_error(
                f"Entity type '{entity.ontology_type}' requires at least "
                f"{emb_req.min_embeddings} embedding(s), "
                f"but entity has {len(entity.embeddings)}"
            )

        # Check required embedding types
        embedding_types = {e.embedding_type.value for e in entity.embeddings}
        for required_type in emb_req.required_types:
            if required_type not in embedding_types:
                result.add_error(
                    f"Entity type '{entity.ontology_type}' requires "
                    f"embedding of type '{required_type}'"
                )

        # Check 8: Global constraints
        for constraint in self.ontology.global_constraints:
            is_valid, error = constraint.validate(entity)
            if not is_valid:
                result.add_error(f"Global constraint failed: {error}")

        return result

    async def validate_relation(
        self,
        source_entity: "KnowledgeEntity",
        relation_type: str,
        target_entity: "KnowledgeEntity",
        properties: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a relation between two entities.

        Checks:
        1. Relation type exists in ontology
        2. Source type is allowed
        3. Target type is allowed
        4. Relation properties are valid
        5. Cardinality constraints (requires storage)
        6. Relation-level constraints

        Args:
            source_entity: The source entity
            relation_type: The relation type name
            target_entity: The target entity
            properties: Optional properties on the relation

        Returns:
            ValidationResult with validation status and error messages
        """
        result = ValidationResult(valid=True)

        # Check 1: Relation type exists
        if relation_type not in self.ontology.relation_types:
            result.add_error(
                f"Unknown relation type: '{relation_type}'. "
                f"Available types: {list(self.ontology.relation_types.keys())}"
            )
            return result

        rel_def = self.ontology.relation_types[relation_type]

        # Check 2: Source type is allowed
        if not rel_def.allows_source_type(source_entity.ontology_type, self.ontology):
            result.add_error(
                f"Relation '{relation_type}' does not allow source type "
                f"'{source_entity.ontology_type}'. "
                f"Allowed types: {rel_def.allowed_source_types}"
            )

        # Check 3: Target type is allowed
        if not rel_def.allows_target_type(target_entity.ontology_type, self.ontology):
            result.add_error(
                f"Relation '{relation_type}' does not allow target type "
                f"'{target_entity.ontology_type}'. "
                f"Allowed types: {rel_def.allowed_target_types}"
            )

        # Check 4: Relation properties
        properties = properties or {}

        # Check required properties
        for prop_name in rel_def.required_properties:
            if prop_name not in properties or properties[prop_name] is None:
                result.add_error(
                    f"Missing required property '{prop_name}' "
                    f"for relation '{relation_type}'"
                )

        # Validate property values
        for prop_name, value in properties.items():
            if prop_name in rel_def.properties:
                prop_def = rel_def.properties[prop_name]
                is_valid, errors = prop_def.validate_value(value)
                if not is_valid:
                    for error in errors:
                        result.add_error(error)
            else:
                result.add_warning(
                    f"Unknown property '{prop_name}' for relation '{relation_type}'"
                )

        # Check 5: Cardinality constraints
        # Note: This requires storage access to count existing relations
        if self.storage:
            cardinality_result = await self._validate_cardinality(
                source_entity, relation_type, target_entity, rel_def
            )
            result.merge(cardinality_result)

        # Check 6: Relation-level constraints
        for constraint in rel_def.constraints:
            # Create a pseudo-object with relation data for validation
            relation_data = {
                "source": source_entity,
                "target": target_entity,
                "type": relation_type,
                "properties": properties
            }
            is_valid, error = constraint.validate(relation_data)
            if not is_valid:
                result.add_error(
                    f"Relation constraint failed for '{relation_type}': {error}"
                )

        return result

    async def _validate_cardinality(
        self,
        source_entity: "KnowledgeEntity",
        relation_type: str,
        target_entity: "KnowledgeEntity",
        rel_def: RelationTypeDefinition
    ) -> ValidationResult:
        """
        Validate cardinality constraints.

        This requires storage access to count existing relations.
        """
        result = ValidationResult(valid=True)

        if not self.storage:
            return result

        try:
            # Check source cardinality
            # Count how many outgoing relations of this type the source already has
            outgoing_count = await self._count_outgoing_relations(
                source_entity.id, relation_type
            )

            if rel_def.source_cardinality == Cardinality.ONE:
                if outgoing_count >= 1:
                    result.add_error(
                        f"Source entity can only have ONE outgoing '{relation_type}' "
                        f"relation, but already has {outgoing_count}"
                    )
            elif rel_def.source_cardinality == Cardinality.ZERO_OR_ONE:
                if outgoing_count >= 1:
                    result.add_error(
                        f"Source entity can have at most ONE outgoing '{relation_type}' "
                        f"relation, but already has {outgoing_count}"
                    )

            # Check target cardinality
            # Count how many incoming relations of this type the target already has
            incoming_count = await self._count_incoming_relations(
                target_entity.id, relation_type
            )

            if rel_def.target_cardinality == Cardinality.ONE:
                if incoming_count >= 1:
                    result.add_error(
                        f"Target entity can only have ONE incoming '{relation_type}' "
                        f"relation, but already has {incoming_count}"
                    )
            elif rel_def.target_cardinality == Cardinality.ZERO_OR_ONE:
                if incoming_count >= 1:
                    result.add_error(
                        f"Target entity can have at most ONE incoming '{relation_type}' "
                        f"relation, but already has {incoming_count}"
                    )

        except Exception as e:
            result.add_warning(
                f"Could not validate cardinality constraints: {str(e)}"
            )

        return result

    async def _count_outgoing_relations(
        self,
        entity_id: UUID,
        relation_type: str
    ) -> int:
        """Count outgoing relations of a specific type from an entity"""
        if not self.storage:
            return 0

        try:
            relations = await self.storage.get_relations(
                entity_id=entity_id,
                relation_type=relation_type,
                direction="outgoing"
            )
            return len(relations)
        except:
            return 0

    async def _count_incoming_relations(
        self,
        entity_id: UUID,
        relation_type: str
    ) -> int:
        """Count incoming relations of a specific type to an entity"""
        if not self.storage:
            return 0

        try:
            relations = await self.storage.get_relations(
                entity_id=entity_id,
                relation_type=relation_type,
                direction="incoming"
            )
            return len(relations)
        except:
            return 0

    def validate_ontology_consistency(self) -> ValidationResult:
        """
        Validate that the ontology itself is consistent.

        Checks:
        - No circular inheritance
        - Referenced parent types exist
        - Relation type references valid entity types
        - Inverse relations are properly defined
        """
        result = ValidationResult(valid=True)

        # Check for circular inheritance
        for type_name in self.ontology.entity_types:
            if self._has_circular_inheritance(type_name):
                result.add_error(
                    f"Circular inheritance detected for type '{type_name}'"
                )

        # Check parent types exist
        for type_name, type_def in self.ontology.entity_types.items():
            if type_def.inherits_from:
                for parent in type_def.inherits_from:
                    if parent not in self.ontology.entity_types:
                        result.add_error(
                            f"Entity type '{type_name}' inherits from unknown "
                            f"type '{parent}'"
                        )

        # Check relation type references
        for rel_name, rel_def in self.ontology.relation_types.items():
            # Check allowed source types
            for source_type in rel_def.allowed_source_types:
                if source_type not in self.ontology.entity_types:
                    result.add_error(
                        f"Relation '{rel_name}' references unknown source "
                        f"type '{source_type}'"
                    )

            # Check allowed target types
            for target_type in rel_def.allowed_target_types:
                if target_type not in self.ontology.entity_types:
                    result.add_error(
                        f"Relation '{rel_name}' references unknown target "
                        f"type '{target_type}'"
                    )

            # Check inverse relations
            if rel_def.inverse_of:
                if rel_def.inverse_of not in self.ontology.relation_types:
                    result.add_error(
                        f"Relation '{rel_name}' declares inverse_of "
                        f"'{rel_def.inverse_of}' which does not exist"
                    )

        return result

    def _has_circular_inheritance(
        self,
        type_name: str,
        visited: Optional[set] = None
    ) -> bool:
        """Check if a type has circular inheritance"""
        if visited is None:
            visited = set()

        if type_name in visited:
            return True

        visited.add(type_name)

        if type_name not in self.ontology.entity_types:
            return False

        type_def = self.ontology.entity_types[type_name]
        if type_def.inherits_from:
            for parent in type_def.inherits_from:
                if self._has_circular_inheritance(parent, visited.copy()):
                    return True

        return False


# Need to import at the end to avoid circular imports
from pydantic import BaseModel, Field
from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
