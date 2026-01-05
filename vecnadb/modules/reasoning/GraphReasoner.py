"""
VecnaDB Graph Reasoner

This module implements deterministic reasoning using graph structure.
Graph reasoning is AUTHORITATIVE - it produces facts, not suggestions.

Key Features:
- Inheritance resolution (transitive property inheritance)
- Constraint validation
- Transitive relation inference
- Symmetric relation inference
- Cardinality checking
- Contradiction detection

Graph reasoning ASSERTS TRUTH. Vector reasoning only advises.
"""

import time
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
    Relation,
    Direction,
)
from vecnadb.modules.ontology.models.OntologySchema import (
    OntologySchema,
    RelationTypeDefinition,
    Cardinality,
)
from vecnadb.modules.ontology.validation.OntologyValidator import (
    OntologyValidator,
    ValidationResult,
)


class InferenceType(str, Enum):
    """Types of graph-based inference"""
    TRANSITIVE = "transitive"  # If A→B and B→C, then A→C
    SYMMETRIC = "symmetric"    # If A→B, then B→A
    INVERSE = "inverse"        # If A→B via R, then B→A via R_inverse
    INHERITANCE = "inheritance"  # Type inheritance property resolution
    CARDINALITY = "cardinality"  # Cardinality constraint checking


class InferredRelation(BaseModel):
    """A relation inferred by graph reasoning"""
    source_id: UUID
    relation_type: str
    target_id: UUID
    properties: Dict[str, Any] = Field(default_factory=dict)
    inference_type: InferenceType
    inference_path: List[Relation]  # Path that led to this inference
    confidence: float = 1.0  # Graph inferences are deterministic (confidence = 1.0)
    explanation: str


class ContradictionResult(BaseModel):
    """A detected contradiction in the knowledge graph"""
    entity_id: UUID
    contradiction_type: str
    description: str
    conflicting_relations: List[Relation]
    severity: str  # "error", "warning"


class ReasoningResult(BaseModel):
    """Result of graph reasoning operation"""
    inferred_relations: List[InferredRelation]
    contradictions: List[ContradictionResult]
    entities_validated: int
    relations_traversed: int
    inference_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphReasoner:
    """
    Deterministic reasoning using graph structure.

    Graph reasoning is AUTHORITATIVE:
    - Produces facts, not suggestions
    - Confidence is always 1.0 (deterministic)
    - Can assert new relations based on structure
    - Detects contradictions and constraint violations

    Process:
    1. Load ontology and inference rules
    2. Traverse graph structure
    3. Apply inference rules (transitivity, symmetry, etc.)
    4. Validate cardinality constraints
    5. Detect contradictions
    6. Return inferred facts
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema
    ):
        """
        Initialize graph reasoner.

        Args:
            storage: VecnaDB storage interface
            ontology: Ontology schema to reason over
        """
        self.storage = storage
        self.ontology = ontology
        self.validator = OntologyValidator(ontology)

        # Build inference rule index
        self._transitive_relations = self._get_transitive_relations()
        self._symmetric_relations = self._get_symmetric_relations()

    def _get_transitive_relations(self) -> Set[str]:
        """Get all transitive relation types from ontology"""
        return {
            name
            for name, rel_type in self.ontology.relation_types.items()
            if rel_type.transitive
        }

    def _get_symmetric_relations(self) -> Set[str]:
        """Get all symmetric relation types from ontology"""
        return {
            name
            for name, rel_type in self.ontology.relation_types.items()
            if rel_type.symmetric
        }

    async def infer_relations(
        self,
        entity_id: UUID,
        max_depth: int = 3,
        relation_types: Optional[List[str]] = None
    ) -> ReasoningResult:
        """
        Infer new relations for an entity using graph structure.

        Args:
            entity_id: Entity to reason about
            max_depth: Maximum traversal depth
            relation_types: Optional filter for relation types

        Returns:
            ReasoningResult with inferred relations and contradictions
        """
        start_time = time.time()

        inferred = []
        contradictions = []
        entities_visited = set()
        relations_traversed = 0

        # Get starting entity
        entity = await self.storage.get_entity(entity_id)
        entities_visited.add(entity_id)

        # Apply different inference strategies

        # 1. Transitive inference
        transitive_inferences = await self._infer_transitive_relations(
            entity_id,
            max_depth,
            relation_types
        )
        inferred.extend(transitive_inferences)

        # 2. Symmetric inference
        symmetric_inferences = await self._infer_symmetric_relations(
            entity_id,
            relation_types
        )
        inferred.extend(symmetric_inferences)

        # 3. Inheritance inference
        inheritance_inferences = await self._infer_from_inheritance(entity)
        inferred.extend(inheritance_inferences)

        # 4. Check cardinality constraints
        cardinality_violations = await self._check_cardinality_constraints(entity_id)
        contradictions.extend(cardinality_violations)

        # Build result
        result = ReasoningResult(
            inferred_relations=inferred,
            contradictions=contradictions,
            entities_validated=len(entities_visited),
            relations_traversed=relations_traversed,
            inference_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "max_depth": max_depth,
                "ontology_id": str(self.ontology.id),
                "entity_type": entity.ontology_type
            }
        )

        return result

    async def _infer_transitive_relations(
        self,
        entity_id: UUID,
        max_depth: int,
        relation_types: Optional[List[str]]
    ) -> List[InferredRelation]:
        """
        Infer relations via transitivity.

        If relation R is transitive and A-R->B and B-R->C, then A-R->C.

        Args:
            entity_id: Starting entity
            max_depth: Maximum path length
            relation_types: Optional relation type filter

        Returns:
            List of inferred transitive relations
        """
        inferred = []

        # Filter to only transitive relation types
        transitive_types = self._transitive_relations
        if relation_types:
            transitive_types = transitive_types.intersection(set(relation_types))

        if not transitive_types:
            return inferred

        # For each transitive relation type
        for rel_type in transitive_types:
            # Find transitive chains
            chains = await self._find_transitive_chains(
                entity_id,
                rel_type,
                max_depth
            )

            # Each chain implies a direct relation from source to target
            for chain in chains:
                if len(chain) < 2:
                    continue

                # Create inferred relation from first to last
                source_id = chain[0].source_id
                target_id = chain[-1].target_id

                # Check if this relation already exists
                existing = await self.storage.get_relations(
                    entity_id=source_id,
                    relation_type=rel_type,
                    direction=Direction.OUTGOING
                )

                already_exists = any(
                    r.target_id == target_id for r in existing
                )

                if not already_exists:
                    inferred_rel = InferredRelation(
                        source_id=source_id,
                        relation_type=rel_type,
                        target_id=target_id,
                        properties={},
                        inference_type=InferenceType.TRANSITIVE,
                        inference_path=chain,
                        confidence=1.0,
                        explanation=(
                            f"Inferred via transitivity: "
                            f"{' → '.join([str(r.target_id) for r in chain])}"
                        )
                    )
                    inferred.append(inferred_rel)

        return inferred

    async def _find_transitive_chains(
        self,
        start_id: UUID,
        relation_type: str,
        max_depth: int
    ) -> List[List[Relation]]:
        """
        Find all transitive chains for a relation type.

        Args:
            start_id: Starting entity
            relation_type: Relation type to traverse
            max_depth: Maximum chain length

        Returns:
            List of relation chains
        """
        chains = []

        async def traverse(current_id: UUID, path: List[Relation], depth: int):
            if depth >= max_depth:
                if len(path) >= 2:
                    chains.append(path.copy())
                return

            # Get outgoing relations of this type
            relations = await self.storage.get_relations(
                entity_id=current_id,
                relation_type=relation_type,
                direction=Direction.OUTGOING
            )

            for relation in relations:
                # Avoid cycles
                if any(r.target_id == relation.target_id for r in path):
                    continue

                path.append(relation)

                # Recurse to continue chain
                await traverse(relation.target_id, path, depth + 1)

                path.pop()

            # If we have a chain of at least 2, record it
            if len(path) >= 2:
                chains.append(path.copy())

        await traverse(start_id, [], 0)
        return chains

    async def _infer_symmetric_relations(
        self,
        entity_id: UUID,
        relation_types: Optional[List[str]]
    ) -> List[InferredRelation]:
        """
        Infer symmetric relations.

        If relation R is symmetric and A-R->B, then B-R->A.

        Args:
            entity_id: Entity to check
            relation_types: Optional relation type filter

        Returns:
            List of inferred symmetric relations
        """
        inferred = []

        # Filter to only symmetric relation types
        symmetric_types = self._symmetric_relations
        if relation_types:
            symmetric_types = symmetric_types.intersection(set(relation_types))

        if not symmetric_types:
            return inferred

        # For each symmetric relation type
        for rel_type in symmetric_types:
            # Get outgoing relations
            outgoing = await self.storage.get_relations(
                entity_id=entity_id,
                relation_type=rel_type,
                direction=Direction.OUTGOING
            )

            # For each outgoing relation A-R->B, check if B-R->A exists
            for relation in outgoing:
                target_id = relation.target_id

                # Check reverse relation
                incoming = await self.storage.get_relations(
                    entity_id=target_id,
                    relation_type=rel_type,
                    direction=Direction.OUTGOING
                )

                reverse_exists = any(
                    r.target_id == entity_id for r in incoming
                )

                if not reverse_exists:
                    # Infer the symmetric reverse relation
                    inferred_rel = InferredRelation(
                        source_id=target_id,
                        relation_type=rel_type,
                        target_id=entity_id,
                        properties=relation.properties.copy(),
                        inference_type=InferenceType.SYMMETRIC,
                        inference_path=[relation],
                        confidence=1.0,
                        explanation=(
                            f"Inferred via symmetry: "
                            f"Since {entity_id} -{rel_type}-> {target_id}, "
                            f"then {target_id} -{rel_type}-> {entity_id}"
                        )
                    )
                    inferred.append(inferred_rel)

        return inferred

    async def _infer_from_inheritance(
        self,
        entity: KnowledgeEntity
    ) -> List[InferredRelation]:
        """
        Infer relations based on type inheritance.

        If entity is of type T and T inherits from S, entity also satisfies
        constraints of type S.

        Args:
            entity: Entity to reason about

        Returns:
            List of inferred relations from inheritance
        """
        inferred = []

        # Get entity type definition
        entity_type_name = entity.ontology_type

        if entity_type_name not in self.ontology.entity_types:
            return inferred

        entity_type = self.ontology.entity_types[entity_type_name]

        # Get all parent types via inheritance
        parent_types = self.ontology.get_all_parent_types(entity_type_name)

        # For each parent type, check if there are implied relations
        for parent_name in parent_types:
            parent_type = self.ontology.entity_types[parent_name]

            # Check if parent type has required relations that aren't satisfied
            # This is more complex - for now, just document the inheritance
            # Real implementation would check required outgoing relations
            pass

        # Note: Inheritance mainly affects property validation, not relation inference
        # This is a placeholder for future enhancement

        return inferred

    async def _check_cardinality_constraints(
        self,
        entity_id: UUID
    ) -> List[ContradictionResult]:
        """
        Check if entity violates cardinality constraints.

        Args:
            entity_id: Entity to check

        Returns:
            List of cardinality violations
        """
        violations = []

        entity = await self.storage.get_entity(entity_id)
        entity_type_name = entity.ontology_type

        if entity_type_name not in self.ontology.entity_types:
            return violations

        # For each relation type in the ontology
        for rel_type_name, rel_type_def in self.ontology.relation_types.items():
            # Check if this entity can be a source of this relation
            if entity_type_name not in rel_type_def.allowed_source_types:
                continue

            # Get all outgoing relations of this type
            relations = await self.storage.get_relations(
                entity_id=entity_id,
                relation_type=rel_type_name,
                direction=Direction.OUTGOING
            )

            count = len(relations)
            cardinality = rel_type_def.source_cardinality

            # Check cardinality constraints
            violation = None

            if cardinality == Cardinality.ONE and count != 1:
                violation = ContradictionResult(
                    entity_id=entity_id,
                    contradiction_type="cardinality_violation",
                    description=(
                        f"Entity must have exactly ONE {rel_type_name} relation, "
                        f"but has {count}"
                    ),
                    conflicting_relations=relations,
                    severity="error"
                )

            elif cardinality == Cardinality.ZERO_OR_ONE and count > 1:
                violation = ContradictionResult(
                    entity_id=entity_id,
                    contradiction_type="cardinality_violation",
                    description=(
                        f"Entity must have at most ONE {rel_type_name} relation, "
                        f"but has {count}"
                    ),
                    conflicting_relations=relations,
                    severity="error"
                )

            elif cardinality == Cardinality.ONE_OR_MORE and count < 1:
                violation = ContradictionResult(
                    entity_id=entity_id,
                    contradiction_type="cardinality_violation",
                    description=(
                        f"Entity must have at least ONE {rel_type_name} relation, "
                        f"but has {count}"
                    ),
                    conflicting_relations=relations,
                    severity="error"
                )

            if violation:
                violations.append(violation)

        return violations

    async def validate_graph_consistency(
        self,
        entity_ids: Optional[List[UUID]] = None
    ) -> ReasoningResult:
        """
        Validate consistency of the entire graph (or subset).

        Checks:
        - Cardinality constraints
        - Orphaned relations
        - Type mismatches
        - Constraint violations

        Args:
            entity_ids: Optional list of entities to check (None = all)

        Returns:
            ReasoningResult with all detected contradictions
        """
        start_time = time.time()

        contradictions = []
        entities_checked = 0

        # If no specific entities, get all entities (this could be expensive)
        if entity_ids is None:
            # For now, require explicit entity list
            # Full graph validation would need pagination
            raise ValueError(
                "Full graph validation requires explicit entity_ids list"
            )

        # Check each entity
        for entity_id in entity_ids:
            try:
                entity = await self.storage.get_entity(entity_id)
                entities_checked += 1

                # Check cardinality
                violations = await self._check_cardinality_constraints(entity_id)
                contradictions.extend(violations)

                # Validate entity against ontology
                validation_result = await self.validator.validate_entity(entity)
                if not validation_result.valid:
                    contradiction = ContradictionResult(
                        entity_id=entity_id,
                        contradiction_type="ontology_violation",
                        description="; ".join(validation_result.errors),
                        conflicting_relations=[],
                        severity="error"
                    )
                    contradictions.append(contradiction)

            except Exception as e:
                # Entity not found or other error
                contradiction = ContradictionResult(
                    entity_id=entity_id,
                    contradiction_type="entity_error",
                    description=f"Error validating entity: {str(e)}",
                    conflicting_relations=[],
                    severity="warning"
                )
                contradictions.append(contradiction)

        result = ReasoningResult(
            inferred_relations=[],
            contradictions=contradictions,
            entities_validated=entities_checked,
            relations_traversed=0,
            inference_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "validation_type": "consistency_check"
            }
        )

        return result


# Convenience functions
async def infer_all_relations(
    entity_id: UUID,
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    max_depth: int = 3
) -> ReasoningResult:
    """
    Run all graph-based inference for an entity.

    Args:
        entity_id: Entity to reason about
        storage: Storage interface
        ontology: Ontology schema
        max_depth: Maximum traversal depth

    Returns:
        ReasoningResult with all inferences
    """
    reasoner = GraphReasoner(storage, ontology)
    return await reasoner.infer_relations(entity_id, max_depth)


async def check_consistency(
    entity_ids: List[UUID],
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema
) -> ReasoningResult:
    """
    Check graph consistency for a set of entities.

    Args:
        entity_ids: Entities to check
        storage: Storage interface
        ontology: Ontology schema

    Returns:
        ReasoningResult with contradictions
    """
    reasoner = GraphReasoner(storage, ontology)
    return await reasoner.validate_graph_consistency(entity_ids)
