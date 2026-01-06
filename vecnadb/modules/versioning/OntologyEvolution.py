"""
VecnaDB Ontology Evolution

Manages ontology schema versioning and evolution over time.

Key Features:
1. Ontology versioning (semantic versioning)
2. Schema change tracking
3. Backward compatibility checking
4. Breaking change detection
5. Migration path generation
6. Ontology inheritance/extension

Principle: Ontologies evolve. Migrations must be traceable and reversible.
"""

import time
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from vecnadb.modules.ontology.models.OntologySchema import (
    OntologySchema,
    EntityTypeDefinition,
    RelationTypeDefinition,
    PropertyDefinition,
)


class SchemaChangeType(str, Enum):
    """Type of schema change"""
    # Entity type changes
    ADD_ENTITY_TYPE = "add_entity_type"
    REMOVE_ENTITY_TYPE = "remove_entity_type"
    RENAME_ENTITY_TYPE = "rename_entity_type"
    MODIFY_ENTITY_TYPE = "modify_entity_type"

    # Relation type changes
    ADD_RELATION_TYPE = "add_relation_type"
    REMOVE_RELATION_TYPE = "remove_relation_type"
    RENAME_RELATION_TYPE = "rename_relation_type"
    MODIFY_RELATION_TYPE = "modify_relation_type"

    # Property changes
    ADD_PROPERTY = "add_property"
    REMOVE_PROPERTY = "remove_property"
    RENAME_PROPERTY = "rename_property"
    MODIFY_PROPERTY = "modify_property"

    # Constraint changes
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    MODIFY_CONSTRAINT = "modify_constraint"


class CompatibilityLevel(str, Enum):
    """Compatibility level of schema change"""
    COMPATIBLE = "compatible"  # Fully backward compatible
    COMPATIBLE_WITH_MIGRATION = "compatible_with_migration"  # Needs migration
    BREAKING = "breaking"  # Breaking change


class SchemaChange(BaseModel):
    """A single change to the ontology schema"""
    change_id: UUID = Field(default_factory=uuid4)
    change_type: SchemaChangeType
    compatibility: CompatibilityLevel
    target_path: str  # e.g., "entity_types.Person" or "relation_types.IS_A"
    description: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OntologyVersion(BaseModel):
    """A specific version of an ontology"""
    version_id: UUID
    ontology_id: UUID
    version_string: str  # Semantic version (e.g., "1.2.0")
    major: int
    minor: int
    patch: int
    schema: OntologySchema
    timestamp: datetime
    created_by: Optional[str] = None
    change_log: str
    changes: List[SchemaChange] = Field(default_factory=list)
    supersedes: Optional[UUID] = None  # Previous version ID
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_major_version(self) -> bool:
        """Check if this is a major version (breaking changes)"""
        return any(c.compatibility == CompatibilityLevel.BREAKING for c in self.changes)

    @property
    def is_minor_version(self) -> bool:
        """Check if this is a minor version (new features)"""
        return not self.is_major_version and len(self.changes) > 0


class MigrationStep(BaseModel):
    """A single step in a migration"""
    step_id: UUID = Field(default_factory=uuid4)
    step_type: str  # "transform", "validate", "reindex", etc.
    description: str
    source_change: Optional[UUID] = None  # SchemaChange that triggered this
    reversible: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MigrationPlan(BaseModel):
    """Plan for migrating data between ontology versions"""
    plan_id: UUID = Field(default_factory=uuid4)
    from_version: str
    to_version: str
    steps: List[MigrationStep]
    estimated_time: Optional[float] = None  # Seconds
    requires_downtime: bool = False
    reversible: bool = True
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OntologyEvolution:
    """
    Manages ontology schema evolution and versioning.

    Tracks all changes to ontology schemas and generates migration paths
    for data when schemas change.
    """

    def __init__(
        self,
        version_storage: Optional[Any] = None
    ):
        """
        Initialize ontology evolution.

        Args:
            version_storage: Storage for version history
        """
        self.version_storage = version_storage
        self._version_cache: Dict[UUID, List[OntologyVersion]] = {}

    async def create_version(
        self,
        ontology: OntologySchema,
        changes: List[SchemaChange],
        created_by: Optional[str] = None,
        change_log: Optional[str] = None
    ) -> OntologyVersion:
        """
        Create a new version of an ontology.

        Args:
            ontology: Ontology in new state
            changes: List of changes made
            created_by: Who created this version
            change_log: Description of changes

        Returns:
            OntologyVersion record
        """
        # Get previous version
        history = await self.get_version_history(ontology.id)
        previous_version = history[-1] if history else None

        # Calculate new version number
        if previous_version:
            major, minor, patch = self._calculate_version_bump(
                previous_version,
                changes
            )
        else:
            major, minor, patch = 1, 0, 0

        version_string = f"{major}.{minor}.{patch}"

        # Auto-generate change log if not provided
        if not change_log:
            change_log = self._generate_change_log(changes)

        # Create version record
        version = OntologyVersion(
            version_id=uuid4(),
            ontology_id=ontology.id,
            version_string=version_string,
            major=major,
            minor=minor,
            patch=patch,
            schema=ontology,
            timestamp=datetime.now(),
            created_by=created_by,
            change_log=change_log,
            changes=changes,
            supersedes=previous_version.version_id if previous_version else None,
            metadata={}
        )

        # Store version
        await self._store_version(version)

        # Update cache
        if ontology.id not in self._version_cache:
            self._version_cache[ontology.id] = []
        self._version_cache[ontology.id].append(version)

        return version

    async def get_version_history(
        self,
        ontology_id: UUID
    ) -> List[OntologyVersion]:
        """
        Get complete version history for an ontology.

        Args:
            ontology_id: Ontology ID

        Returns:
            List of OntologyVersion records, sorted by version
        """
        # Check cache
        if ontology_id in self._version_cache:
            return self._version_cache[ontology_id]

        # Load from storage
        versions = await self._load_versions(ontology_id)

        # Sort by version
        versions.sort(key=lambda v: (v.major, v.minor, v.patch))

        # Cache
        self._version_cache[ontology_id] = versions

        return versions

    async def get_version(
        self,
        ontology_id: UUID,
        version_string: str
    ) -> Optional[OntologyVersion]:
        """
        Get a specific version of an ontology.

        Args:
            ontology_id: Ontology ID
            version_string: Version string (e.g., "1.2.0")

        Returns:
            OntologyVersion or None
        """
        history = await self.get_version_history(ontology_id)

        return next(
            (v for v in history if v.version_string == version_string),
            None
        )

    async def get_latest_version(
        self,
        ontology_id: UUID
    ) -> Optional[OntologyVersion]:
        """
        Get latest version of an ontology.

        Args:
            ontology_id: Ontology ID

        Returns:
            Latest OntologyVersion or None
        """
        history = await self.get_version_history(ontology_id)

        return history[-1] if history else None

    async def compare_versions(
        self,
        ontology_id: UUID,
        from_version: str,
        to_version: str
    ) -> List[SchemaChange]:
        """
        Compare two versions and get changes.

        Args:
            ontology_id: Ontology ID
            from_version: Starting version
            to_version: Ending version

        Returns:
            List of SchemaChange records
        """
        history = await self.get_version_history(ontology_id)

        # Find version indices
        from_idx = next(
            (i for i, v in enumerate(history) if v.version_string == from_version),
            None
        )
        to_idx = next(
            (i for i, v in enumerate(history) if v.version_string == to_version),
            None
        )

        if from_idx is None or to_idx is None:
            return []

        # Collect all changes between versions
        all_changes = []
        for version in history[from_idx + 1:to_idx + 1]:
            all_changes.extend(version.changes)

        return all_changes

    async def generate_migration_plan(
        self,
        ontology_id: UUID,
        from_version: str,
        to_version: str
    ) -> MigrationPlan:
        """
        Generate migration plan between two versions.

        Args:
            ontology_id: Ontology ID
            from_version: Source version
            to_version: Target version

        Returns:
            MigrationPlan with steps
        """
        # Get changes between versions
        changes = await self.compare_versions(ontology_id, from_version, to_version)

        # Generate migration steps
        steps = []
        warnings = []
        requires_downtime = False
        reversible = True

        for change in changes:
            migration_step = self._generate_migration_step(change)
            steps.append(migration_step)

            # Check if this change requires downtime
            if change.compatibility == CompatibilityLevel.BREAKING:
                requires_downtime = True
                warnings.append(
                    f"Breaking change: {change.description}"
                )

            # Check reversibility
            if not migration_step.reversible:
                reversible = False
                warnings.append(
                    f"Irreversible step: {migration_step.description}"
                )

        # Estimate migration time (simplified)
        estimated_time = len(steps) * 10.0  # 10 seconds per step (placeholder)

        plan = MigrationPlan(
            from_version=from_version,
            to_version=to_version,
            steps=steps,
            estimated_time=estimated_time,
            requires_downtime=requires_downtime,
            reversible=reversible,
            warnings=warnings,
            metadata={
                "total_changes": len(changes),
                "breaking_changes": sum(
                    1 for c in changes
                    if c.compatibility == CompatibilityLevel.BREAKING
                )
            }
        )

        return plan

    async def check_compatibility(
        self,
        old_schema: OntologySchema,
        new_schema: OntologySchema
    ) -> Tuple[CompatibilityLevel, List[SchemaChange]]:
        """
        Check compatibility between two schemas.

        Args:
            old_schema: Old ontology schema
            new_schema: New ontology schema

        Returns:
            Tuple of (compatibility level, list of changes)
        """
        changes = self._detect_schema_changes(old_schema, new_schema)

        # Determine overall compatibility
        if any(c.compatibility == CompatibilityLevel.BREAKING for c in changes):
            overall_compatibility = CompatibilityLevel.BREAKING
        elif any(c.compatibility == CompatibilityLevel.COMPATIBLE_WITH_MIGRATION for c in changes):
            overall_compatibility = CompatibilityLevel.COMPATIBLE_WITH_MIGRATION
        else:
            overall_compatibility = CompatibilityLevel.COMPATIBLE

        return overall_compatibility, changes

    def _detect_schema_changes(
        self,
        old_schema: OntologySchema,
        new_schema: OntologySchema
    ) -> List[SchemaChange]:
        """Detect changes between two schemas"""
        changes = []

        # Detect entity type changes
        changes.extend(self._detect_entity_type_changes(old_schema, new_schema))

        # Detect relation type changes
        changes.extend(self._detect_relation_type_changes(old_schema, new_schema))

        return changes

    def _detect_entity_type_changes(
        self,
        old_schema: OntologySchema,
        new_schema: OntologySchema
    ) -> List[SchemaChange]:
        """Detect changes to entity types"""
        changes = []

        old_types = set(old_schema.entity_types.keys())
        new_types = set(new_schema.entity_types.keys())

        # Added entity types
        for type_name in new_types - old_types:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.ADD_ENTITY_TYPE,
                compatibility=CompatibilityLevel.COMPATIBLE,
                target_path=f"entity_types.{type_name}",
                description=f"Added entity type '{type_name}'",
                new_value=new_schema.entity_types[type_name].dict()
            ))

        # Removed entity types
        for type_name in old_types - new_types:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.REMOVE_ENTITY_TYPE,
                compatibility=CompatibilityLevel.BREAKING,
                target_path=f"entity_types.{type_name}",
                description=f"Removed entity type '{type_name}'",
                old_value=old_schema.entity_types[type_name].dict()
            ))

        # Modified entity types
        for type_name in old_types.intersection(new_types):
            old_type = old_schema.entity_types[type_name]
            new_type = new_schema.entity_types[type_name]

            type_changes = self._detect_entity_type_modifications(
                type_name,
                old_type,
                new_type
            )
            changes.extend(type_changes)

        return changes

    def _detect_entity_type_modifications(
        self,
        type_name: str,
        old_type: EntityTypeDefinition,
        new_type: EntityTypeDefinition
    ) -> List[SchemaChange]:
        """Detect modifications to a single entity type"""
        changes = []

        # Check property changes
        old_props = set(old_type.properties.keys())
        new_props = set(new_type.properties.keys())

        # Added properties
        for prop_name in new_props - old_props:
            # Adding non-required property is compatible
            is_required = prop_name in new_type.required_properties
            compatibility = (
                CompatibilityLevel.COMPATIBLE_WITH_MIGRATION
                if is_required
                else CompatibilityLevel.COMPATIBLE
            )

            changes.append(SchemaChange(
                change_type=SchemaChangeType.ADD_PROPERTY,
                compatibility=compatibility,
                target_path=f"entity_types.{type_name}.properties.{prop_name}",
                description=f"Added property '{prop_name}' to '{type_name}'",
                new_value=new_type.properties[prop_name].dict()
            ))

        # Removed properties
        for prop_name in old_props - new_props:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.REMOVE_PROPERTY,
                compatibility=CompatibilityLevel.BREAKING,
                target_path=f"entity_types.{type_name}.properties.{prop_name}",
                description=f"Removed property '{prop_name}' from '{type_name}'",
                old_value=old_type.properties[prop_name].dict()
            ))

        # Check required properties changes
        old_required = set(old_type.required_properties)
        new_required = set(new_type.required_properties)

        # Newly required properties
        for prop_name in new_required - old_required:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.MODIFY_PROPERTY,
                compatibility=CompatibilityLevel.BREAKING,
                target_path=f"entity_types.{type_name}.required_properties",
                description=f"Property '{prop_name}' is now required in '{type_name}'",
                old_value=False,
                new_value=True
            ))

        return changes

    def _detect_relation_type_changes(
        self,
        old_schema: OntologySchema,
        new_schema: OntologySchema
    ) -> List[SchemaChange]:
        """Detect changes to relation types"""
        changes = []

        old_rels = set(old_schema.relation_types.keys())
        new_rels = set(new_schema.relation_types.keys())

        # Added relation types
        for rel_name in new_rels - old_rels:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.ADD_RELATION_TYPE,
                compatibility=CompatibilityLevel.COMPATIBLE,
                target_path=f"relation_types.{rel_name}",
                description=f"Added relation type '{rel_name}'",
                new_value=new_schema.relation_types[rel_name].dict()
            ))

        # Removed relation types
        for rel_name in old_rels - new_rels:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.REMOVE_RELATION_TYPE,
                compatibility=CompatibilityLevel.BREAKING,
                target_path=f"relation_types.{rel_name}",
                description=f"Removed relation type '{rel_name}'",
                old_value=old_schema.relation_types[rel_name].dict()
            ))

        return changes

    def _calculate_version_bump(
        self,
        previous_version: OntologyVersion,
        changes: List[SchemaChange]
    ) -> Tuple[int, int, int]:
        """
        Calculate version bump based on changes.

        Follows semantic versioning:
        - MAJOR: Breaking changes
        - MINOR: New features (backward compatible)
        - PATCH: Bug fixes (backward compatible)
        """
        has_breaking = any(
            c.compatibility == CompatibilityLevel.BREAKING
            for c in changes
        )

        has_features = any(
            c.change_type in {
                SchemaChangeType.ADD_ENTITY_TYPE,
                SchemaChangeType.ADD_RELATION_TYPE,
                SchemaChangeType.ADD_PROPERTY
            }
            for c in changes
        )

        major = previous_version.major
        minor = previous_version.minor
        patch = previous_version.patch

        if has_breaking:
            # Major version bump
            return major + 1, 0, 0
        elif has_features:
            # Minor version bump
            return major, minor + 1, 0
        else:
            # Patch version bump
            return major, minor, patch + 1

    def _generate_change_log(self, changes: List[SchemaChange]) -> str:
        """Generate change log from changes"""
        if not changes:
            return "No changes"

        lines = []

        # Group by change type
        by_type: Dict[SchemaChangeType, List[SchemaChange]] = {}
        for change in changes:
            if change.change_type not in by_type:
                by_type[change.change_type] = []
            by_type[change.change_type].append(change)

        # Format each group
        for change_type, type_changes in by_type.items():
            lines.append(f"\n{change_type.value}:")
            for change in type_changes:
                lines.append(f"  - {change.description}")

        return "\n".join(lines)

    def _generate_migration_step(self, change: SchemaChange) -> MigrationStep:
        """Generate migration step for a schema change"""
        if change.change_type == SchemaChangeType.ADD_PROPERTY:
            return MigrationStep(
                step_type="add_property",
                description=f"Add default values for new property",
                source_change=change.change_id,
                reversible=True
            )

        elif change.change_type == SchemaChangeType.REMOVE_PROPERTY:
            return MigrationStep(
                step_type="remove_property",
                description=f"Archive old property values",
                source_change=change.change_id,
                reversible=True
            )

        elif change.change_type == SchemaChangeType.RENAME_PROPERTY:
            return MigrationStep(
                step_type="rename_property",
                description=f"Rename property in all entities",
                source_change=change.change_id,
                reversible=True
            )

        else:
            return MigrationStep(
                step_type="generic",
                description=f"Handle {change.change_type.value}",
                source_change=change.change_id,
                reversible=False
            )

    async def _store_version(self, version: OntologyVersion):
        """Store version record (placeholder)"""
        # Actual implementation would store to database
        pass

    async def _load_versions(self, ontology_id: UUID) -> List[OntologyVersion]:
        """Load version history (placeholder)"""
        # Actual implementation would load from database
        return []


# Convenience functions
async def evolve_ontology(
    old_schema: OntologySchema,
    new_schema: OntologySchema,
    created_by: Optional[str] = None
) -> Tuple[OntologyVersion, CompatibilityLevel]:
    """
    Evolve an ontology to a new schema.

    Args:
        old_schema: Current schema
        new_schema: New schema
        created_by: Who is making the change

    Returns:
        Tuple of (new version, compatibility level)
    """
    evolution = OntologyEvolution()

    # Check compatibility
    compatibility, changes = await evolution.check_compatibility(old_schema, new_schema)

    # Create new version
    version = await evolution.create_version(
        ontology=new_schema,
        changes=changes,
        created_by=created_by
    )

    return version, compatibility


async def plan_migration(
    ontology_id: UUID,
    from_version: str,
    to_version: str
) -> MigrationPlan:
    """
    Plan migration between ontology versions.

    Args:
        ontology_id: Ontology ID
        from_version: Source version
        to_version: Target version

    Returns:
        MigrationPlan
    """
    evolution = OntologyEvolution()
    return await evolution.generate_migration_plan(ontology_id, from_version, to_version)
