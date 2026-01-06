"""
VecnaDB Context Validator

Validates retrieved context for RAG to ensure quality and ontology compliance.

Validation Checks:
1. Ontology compliance (entity types, properties, constraints)
2. Dual representation (graph + vector data)
3. Relation consistency
4. Temporal validity
5. Content quality
6. Relevance to query

Only ontology-valid, high-quality context should be used for answer generation.
"""

import time
from typing import List, Optional, Dict, Any, Set
from uuid import UUID
from enum import Enum
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
    Relation,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema
from vecnadb.modules.ontology.validation.OntologyValidator import OntologyValidator


class ValidationLevel(str, Enum):
    """Validation strictness level"""
    STRICT = "strict"  # All checks must pass
    MODERATE = "moderate"  # Most checks must pass
    LENIENT = "lenient"  # Basic checks only


class ValidationIssue(str, Enum):
    """Types of validation issues"""
    ONTOLOGY_VIOLATION = "ontology_violation"
    MISSING_EMBEDDINGS = "missing_embeddings"
    MISSING_GRAPH_NODE = "missing_graph_node"
    STALE_DATA = "stale_data"
    LOW_QUALITY_CONTENT = "low_quality_content"
    INCONSISTENT_RELATIONS = "inconsistent_relations"
    INVALID_PROPERTIES = "invalid_properties"


class ValidationResult(BaseModel):
    """Result of context validation"""
    entity_id: UUID
    is_valid: bool
    validity_score: float  # 0.0-1.0
    issues: List[ValidationIssue] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextValidationReport(BaseModel):
    """Complete validation report for context set"""
    total_entities: int
    valid_entities: int
    invalid_entities: int
    avg_validity_score: float
    validation_level: ValidationLevel
    entity_results: List[ValidationResult]
    overall_valid: bool
    validation_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_valid_entities(self) -> List[UUID]:
        """Get list of valid entity IDs"""
        return [
            result.entity_id
            for result in self.entity_results
            if result.is_valid
        ]

    def get_invalid_entities(self) -> List[UUID]:
        """Get list of invalid entity IDs"""
        return [
            result.entity_id
            for result in self.entity_results
            if not result.is_valid
        ]

    def get_issues_by_type(self, issue_type: ValidationIssue) -> List[ValidationResult]:
        """Get all results with specific issue type"""
        return [
            result
            for result in self.entity_results
            if issue_type in result.issues
        ]


class ContextValidator:
    """
    Validates retrieved context for RAG quality and compliance.

    Ensures:
    - All entities are ontology-compliant
    - Dual representation is intact
    - Content is high quality
    - Relations are consistent
    - Data is not stale
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema,
        max_staleness_days: int = 365
    ):
        """
        Initialize context validator.

        Args:
            storage: VecnaDB storage interface
            ontology: Ontology schema
            max_staleness_days: Maximum age for data
        """
        self.storage = storage
        self.ontology = ontology
        self.max_staleness_days = max_staleness_days
        self.ontology_validator = OntologyValidator(ontology)

    async def validate_entity(
        self,
        entity: KnowledgeEntity,
        level: ValidationLevel = ValidationLevel.MODERATE
    ) -> ValidationResult:
        """
        Validate a single entity for RAG context use.

        Args:
            entity: Entity to validate
            level: Validation strictness level

        Returns:
            ValidationResult with issues and score
        """
        issues = []
        warnings = []
        checks_passed = 0
        total_checks = 0

        # 1. Ontology compliance
        total_checks += 1
        ontology_result = await self.ontology_validator.validate_entity(entity)
        if not ontology_result.valid:
            issues.append(ValidationIssue.ONTOLOGY_VIOLATION)
            warnings.append(f"Ontology violations: {'; '.join(ontology_result.errors)}")
        else:
            checks_passed += 1

        # 2. Dual representation check
        total_checks += 1
        if not entity.embeddings or len(entity.embeddings) == 0:
            issues.append(ValidationIssue.MISSING_EMBEDDINGS)
            warnings.append("Entity missing embeddings")
        elif not entity.graph_node_id:
            issues.append(ValidationIssue.MISSING_GRAPH_NODE)
            warnings.append("Entity missing graph node ID")
        else:
            checks_passed += 1

        # 3. Temporal validity (staleness check)
        total_checks += 1
        if self._is_stale(entity):
            issues.append(ValidationIssue.STALE_DATA)
            warnings.append(f"Entity older than {self.max_staleness_days} days")
        else:
            checks_passed += 1

        # 4. Content quality (for STRICT and MODERATE)
        if level in [ValidationLevel.STRICT, ValidationLevel.MODERATE]:
            total_checks += 1
            if not self._has_quality_content(entity):
                issues.append(ValidationIssue.LOW_QUALITY_CONTENT)
                warnings.append("Low quality or missing content")
            else:
                checks_passed += 1

        # 5. Property validation (for STRICT)
        if level == ValidationLevel.STRICT:
            total_checks += 1
            if not self._validate_properties(entity):
                issues.append(ValidationIssue.INVALID_PROPERTIES)
                warnings.append("Invalid or inconsistent properties")
            else:
                checks_passed += 1

        # Calculate validity score
        validity_score = checks_passed / total_checks if total_checks > 0 else 0.0

        # Determine if valid based on level
        is_valid = self._determine_validity(level, validity_score, issues)

        result = ValidationResult(
            entity_id=entity.id,
            is_valid=is_valid,
            validity_score=validity_score,
            issues=issues,
            warnings=warnings,
            metadata={
                "checks_passed": checks_passed,
                "total_checks": total_checks,
                "entity_type": entity.ontology_type,
                "validation_level": level
            }
        )

        return result

    async def validate_context(
        self,
        entities: List[KnowledgeEntity],
        level: ValidationLevel = ValidationLevel.MODERATE,
        min_valid_ratio: float = 0.8
    ) -> ContextValidationReport:
        """
        Validate a set of context entities.

        Args:
            entities: Entities to validate
            level: Validation strictness
            min_valid_ratio: Minimum ratio of valid entities

        Returns:
            ContextValidationReport with detailed results
        """
        start_time = time.time()

        entity_results = []

        # Validate each entity
        for entity in entities:
            result = await self.validate_entity(entity, level)
            entity_results.append(result)

        # Calculate aggregates
        valid_count = sum(1 for r in entity_results if r.is_valid)
        invalid_count = len(entity_results) - valid_count

        avg_validity = (
            sum(r.validity_score for r in entity_results) / len(entity_results)
            if entity_results else 0.0
        )

        valid_ratio = valid_count / len(entity_results) if entity_results else 0.0
        overall_valid = valid_ratio >= min_valid_ratio

        report = ContextValidationReport(
            total_entities=len(entities),
            valid_entities=valid_count,
            invalid_entities=invalid_count,
            avg_validity_score=avg_validity,
            validation_level=level,
            entity_results=entity_results,
            overall_valid=overall_valid,
            validation_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "min_valid_ratio": min_valid_ratio,
                "actual_valid_ratio": valid_ratio
            }
        )

        return report

    async def validate_relations(
        self,
        entity_id: UUID,
        relation_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate relations for an entity.

        Args:
            entity_id: Entity to check relations for
            relation_type: Optional specific relation type

        Returns:
            ValidationResult for relations
        """
        issues = []
        warnings = []

        try:
            entity = await self.storage.get_entity(entity_id)

            # Get all relations
            outgoing = await self.storage.get_relations(
                entity_id=entity_id,
                relation_type=relation_type,
                direction="OUTGOING"
            )

            incoming = await self.storage.get_relations(
                entity_id=entity_id,
                relation_type=relation_type,
                direction="INCOMING"
            )

            # Check each outgoing relation
            for relation in outgoing:
                # Validate relation type exists in ontology
                if relation.relation_type not in self.ontology.relation_types:
                    issues.append(ValidationIssue.INCONSISTENT_RELATIONS)
                    warnings.append(f"Unknown relation type: {relation.relation_type}")
                    continue

                # Validate source and target types
                rel_type_def = self.ontology.relation_types[relation.relation_type]

                if entity.ontology_type not in rel_type_def.allowed_source_types:
                    issues.append(ValidationIssue.INCONSISTENT_RELATIONS)
                    warnings.append(
                        f"Entity type '{entity.ontology_type}' not allowed as "
                        f"source for '{relation.relation_type}'"
                    )

                # Check if target exists
                try:
                    target = await self.storage.get_entity(relation.target_id)
                    if target.ontology_type not in rel_type_def.allowed_target_types:
                        issues.append(ValidationIssue.INCONSISTENT_RELATIONS)
                        warnings.append(
                            f"Target type '{target.ontology_type}' not allowed for "
                            f"'{relation.relation_type}'"
                        )
                except Exception:
                    issues.append(ValidationIssue.INCONSISTENT_RELATIONS)
                    warnings.append(f"Target entity not found: {relation.target_id}")

            is_valid = len(issues) == 0
            validity_score = 1.0 if is_valid else 0.5

            result = ValidationResult(
                entity_id=entity_id,
                is_valid=is_valid,
                validity_score=validity_score,
                issues=issues,
                warnings=warnings,
                metadata={
                    "outgoing_relations": len(outgoing),
                    "incoming_relations": len(incoming)
                }
            )

            return result

        except Exception as e:
            return ValidationResult(
                entity_id=entity_id,
                is_valid=False,
                validity_score=0.0,
                issues=[ValidationIssue.INCONSISTENT_RELATIONS],
                warnings=[f"Error validating relations: {str(e)}"],
                metadata={}
            )

    def _is_stale(self, entity: KnowledgeEntity) -> bool:
        """Check if entity data is stale"""
        if not entity.updated_at:
            return False  # Can't determine, assume fresh

        age = datetime.now() - entity.updated_at
        return age.days > self.max_staleness_days

    def _has_quality_content(self, entity: KnowledgeEntity) -> bool:
        """Check if entity has quality content"""
        # Check for required properties
        if not entity.properties:
            return False

        # Check for name or title
        has_identifier = (
            'name' in entity.properties or
            'title' in entity.properties
        )

        if not has_identifier:
            return False

        # Check for content
        content = entity.properties.get('content', '')
        if isinstance(content, str):
            # Content should be non-empty and substantial
            return len(content.strip()) > 10

        return True

    def _validate_properties(self, entity: KnowledgeEntity) -> bool:
        """Validate entity properties against ontology"""
        if entity.ontology_type not in self.ontology.entity_types:
            return False

        entity_type_def = self.ontology.entity_types[entity.ontology_type]

        # Check required properties
        for required_prop in entity_type_def.required_properties:
            if required_prop not in entity.properties:
                return False

        # Check property types (basic validation)
        for prop_name, prop_value in entity.properties.items():
            if prop_name in entity_type_def.properties:
                prop_def = entity_type_def.properties[prop_name]
                # Would validate type here (string, int, etc.)
                # Simplified for now
                pass

        return True

    def _determine_validity(
        self,
        level: ValidationLevel,
        validity_score: float,
        issues: List[ValidationIssue]
    ) -> bool:
        """Determine if entity is valid based on level and score"""
        if level == ValidationLevel.STRICT:
            # All checks must pass
            return validity_score == 1.0 and len(issues) == 0

        elif level == ValidationLevel.MODERATE:
            # Most checks must pass, no critical issues
            critical_issues = {
                ValidationIssue.ONTOLOGY_VIOLATION,
                ValidationIssue.MISSING_EMBEDDINGS,
                ValidationIssue.MISSING_GRAPH_NODE
            }

            has_critical = any(issue in critical_issues for issue in issues)
            return not has_critical and validity_score >= 0.7

        else:  # LENIENT
            # Basic checks only
            return validity_score >= 0.5

    async def filter_valid_entities(
        self,
        entities: List[KnowledgeEntity],
        level: ValidationLevel = ValidationLevel.MODERATE
    ) -> List[KnowledgeEntity]:
        """
        Filter entities to only valid ones.

        Args:
            entities: Entities to filter
            level: Validation level

        Returns:
            List of valid entities only
        """
        valid_entities = []

        for entity in entities:
            result = await self.validate_entity(entity, level)
            if result.is_valid:
                valid_entities.append(entity)

        return valid_entities


# Convenience functions
async def validate_rag_context(
    entities: List[KnowledgeEntity],
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    level: ValidationLevel = ValidationLevel.MODERATE
) -> ContextValidationReport:
    """
    Validate context for RAG use.

    Args:
        entities: Context entities to validate
        storage: Storage interface
        ontology: Ontology schema
        level: Validation level

    Returns:
        ContextValidationReport
    """
    validator = ContextValidator(storage, ontology)
    return await validator.validate_context(entities, level)


async def filter_invalid_context(
    entities: List[KnowledgeEntity],
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    level: ValidationLevel = ValidationLevel.MODERATE
) -> List[KnowledgeEntity]:
    """
    Remove invalid entities from context.

    Args:
        entities: Context entities
        storage: Storage interface
        ontology: Ontology schema
        level: Validation level

    Returns:
        Filtered list of valid entities
    """
    validator = ContextValidator(storage, ontology)
    return await validator.filter_valid_entities(entities, level)
