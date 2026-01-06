"""
VecnaDB Migration Tools

Tools for migrating data between ontology versions.

Key Features:
1. Execute migration plans
2. Transform entities for new schema
3. Validate migrated data
4. Rollback failed migrations
5. Progress tracking
6. Dry-run mode

Principle: Migrations must be safe, traceable, and reversible.
"""

import time
import asyncio
from typing import List, Optional, Dict, Any, Set, Callable
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema
from vecnadb.modules.ontology.validation.OntologyValidator import OntologyValidator
from vecnadb.modules.versioning.OntologyEvolution import (
    MigrationPlan,
    MigrationStep,
    SchemaChange,
    SchemaChangeType,
)
from vecnadb.modules.versioning.EntityVersioning import (
    EntityVersioning,
    ChangeType,
    ChangeSource,
)


class MigrationStatus(str, Enum):
    """Status of migration execution"""
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationResult(BaseModel):
    """Result of migration execution"""
    migration_id: UUID
    status: MigrationStatus
    entities_migrated: int
    entities_failed: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MigrationProgress(BaseModel):
    """Progress tracking for migration"""
    total_entities: int
    processed_entities: int
    successful: int
    failed: int
    current_step: Optional[str] = None
    percent_complete: float = 0.0
    estimated_time_remaining: Optional[float] = None


class MigrationExecutor:
    """
    Executes migration plans to transform data between ontology versions.

    Ensures data integrity and provides rollback capability for failed migrations.
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        source_ontology: OntologySchema,
        target_ontology: OntologySchema
    ):
        """
        Initialize migration executor.

        Args:
            storage: VecnaDB storage interface
            source_ontology: Current ontology schema
            target_ontology: Target ontology schema
        """
        self.storage = storage
        self.source_ontology = source_ontology
        self.target_ontology = target_ontology
        self.source_validator = OntologyValidator(source_ontology)
        self.target_validator = OntologyValidator(target_ontology)
        self.versioning = EntityVersioning(storage)

    async def execute_migration(
        self,
        plan: MigrationPlan,
        dry_run: bool = False,
        batch_size: int = 100,
        progress_callback: Optional[Callable[[MigrationProgress], None]] = None
    ) -> MigrationResult:
        """
        Execute a migration plan.

        Args:
            plan: Migration plan to execute
            dry_run: If True, validate but don't commit changes
            batch_size: Number of entities to process per batch
            progress_callback: Optional callback for progress updates

        Returns:
            MigrationResult with execution details
        """
        migration_id = uuid4()
        start_time = datetime.now()

        result = MigrationResult(
            migration_id=migration_id,
            status=MigrationStatus.RUNNING,
            entities_migrated=0,
            entities_failed=0,
            start_time=start_time,
            metadata={
                "dry_run": dry_run,
                "plan_id": str(plan.plan_id),
                "from_version": plan.from_version,
                "to_version": plan.to_version
            }
        )

        try:
            # Get all entities to migrate
            entities = await self._get_entities_to_migrate()

            total_entities = len(entities)
            processed = 0

            # Process in batches
            for i in range(0, total_entities, batch_size):
                batch = entities[i:i + batch_size]

                # Process batch
                batch_results = await self._process_batch(
                    batch,
                    plan,
                    dry_run
                )

                # Update counts
                result.entities_migrated += sum(1 for r in batch_results if r)
                result.entities_failed += sum(1 for r in batch_results if not r)
                processed += len(batch)

                # Report progress
                if progress_callback:
                    progress = MigrationProgress(
                        total_entities=total_entities,
                        processed_entities=processed,
                        successful=result.entities_migrated,
                        failed=result.entities_failed,
                        current_step=f"Processing batch {i // batch_size + 1}",
                        percent_complete=(processed / total_entities) * 100
                    )
                    progress_callback(progress)

            # Validation phase
            result.status = MigrationStatus.VALIDATING

            if not dry_run:
                validation_errors = await self._validate_migrated_data()
                if validation_errors:
                    result.errors.extend(validation_errors)
                    result.status = MigrationStatus.FAILED
                else:
                    result.status = MigrationStatus.COMPLETED

            else:
                result.status = MigrationStatus.COMPLETED
                result.warnings.append("Dry run - no changes committed")

            # Calculate duration
            result.end_time = datetime.now()
            result.duration_seconds = (
                result.end_time - result.start_time
            ).total_seconds()

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.errors.append(f"Migration failed: {str(e)}")
            result.end_time = datetime.now()
            result.duration_seconds = (
                result.end_time - result.start_time
            ).total_seconds()

        return result

    async def rollback_migration(
        self,
        migration_id: UUID
    ) -> MigrationResult:
        """
        Rollback a completed migration.

        Args:
            migration_id: ID of migration to rollback

        Returns:
            MigrationResult for rollback operation
        """
        # Placeholder - actual implementation would:
        # 1. Get list of entities modified in migration
        # 2. Restore previous versions
        # 3. Validate restored data

        result = MigrationResult(
            migration_id=migration_id,
            status=MigrationStatus.ROLLED_BACK,
            entities_migrated=0,
            entities_failed=0,
            start_time=datetime.now(),
            metadata={"rollback_of": str(migration_id)}
        )

        return result

    async def _get_entities_to_migrate(self) -> List[KnowledgeEntity]:
        """Get all entities that need migration"""
        # Placeholder - actual implementation would query storage
        # for all entities using the old ontology
        return []

    async def _process_batch(
        self,
        entities: List[KnowledgeEntity],
        plan: MigrationPlan,
        dry_run: bool
    ) -> List[bool]:
        """Process a batch of entities"""
        results = []

        for entity in entities:
            try:
                # Transform entity
                migrated_entity = await self._transform_entity(entity, plan)

                # Validate
                validation = await self.target_validator.validate_entity(migrated_entity)

                if not validation.valid:
                    results.append(False)
                    continue

                # Commit if not dry run
                if not dry_run:
                    await self._commit_migrated_entity(entity, migrated_entity)

                results.append(True)

            except Exception:
                results.append(False)

        return results

    async def _transform_entity(
        self,
        entity: KnowledgeEntity,
        plan: MigrationPlan
    ) -> KnowledgeEntity:
        """
        Transform entity according to migration plan.

        Args:
            entity: Entity in old schema
            plan: Migration plan

        Returns:
            Entity in new schema
        """
        # Create a copy to transform
        transformed = entity.copy(deep=True)

        # Apply each migration step
        for step in plan.steps:
            if step.step_type == "add_property":
                transformed = self._apply_add_property(transformed, step)
            elif step.step_type == "remove_property":
                transformed = self._apply_remove_property(transformed, step)
            elif step.step_type == "rename_property":
                transformed = self._apply_rename_property(transformed, step)
            # ... other step types

        # Update version info
        transformed.version += 1
        transformed.supersedes = entity.id
        transformed.updated_at = datetime.now()

        return transformed

    def _apply_add_property(
        self,
        entity: KnowledgeEntity,
        step: MigrationStep
    ) -> KnowledgeEntity:
        """Apply add property transformation"""
        # Extract property name from step metadata
        prop_name = step.metadata.get("property_name")
        default_value = step.metadata.get("default_value")

        if prop_name and prop_name not in entity.properties:
            entity.properties[prop_name] = default_value

        return entity

    def _apply_remove_property(
        self,
        entity: KnowledgeEntity,
        step: MigrationStep
    ) -> KnowledgeEntity:
        """Apply remove property transformation"""
        prop_name = step.metadata.get("property_name")

        if prop_name and prop_name in entity.properties:
            # Archive old value in metadata
            if "archived_properties" not in entity.properties:
                entity.properties["archived_properties"] = {}
            entity.properties["archived_properties"][prop_name] = entity.properties[prop_name]

            # Remove property
            del entity.properties[prop_name]

        return entity

    def _apply_rename_property(
        self,
        entity: KnowledgeEntity,
        step: MigrationStep
    ) -> KnowledgeEntity:
        """Apply rename property transformation"""
        old_name = step.metadata.get("old_name")
        new_name = step.metadata.get("new_name")

        if old_name and new_name and old_name in entity.properties:
            entity.properties[new_name] = entity.properties[old_name]
            del entity.properties[old_name]

        return entity

    async def _commit_migrated_entity(
        self,
        old_entity: KnowledgeEntity,
        new_entity: KnowledgeEntity
    ):
        """Commit migrated entity to storage"""
        # Store new version
        await self.storage.update_entity(new_entity, validate=True)

        # Track in version history
        await self.versioning.create_version(
            entity=new_entity,
            change_type=ChangeType.UPDATE,
            change_source=ChangeSource.MIGRATION,
            change_description="Migrated to new ontology version",
            previous_entity=old_entity
        )

    async def _validate_migrated_data(self) -> List[str]:
        """Validate all migrated data"""
        errors = []

        # Placeholder - actual implementation would:
        # 1. Get all entities
        # 2. Validate each against new ontology
        # 3. Check dual representation
        # 4. Verify relations are still valid

        return errors


class MigrationBuilder:
    """
    Builder for creating custom migration plans.

    Provides fluent API for defining migrations.
    """

    def __init__(
        self,
        from_version: str,
        to_version: str
    ):
        """
        Initialize migration builder.

        Args:
            from_version: Source version
            to_version: Target version
        """
        self.from_version = from_version
        self.to_version = to_version
        self.steps: List[MigrationStep] = []
        self.warnings: List[str] = []
        self.requires_downtime = False

    def add_property(
        self,
        entity_type: str,
        property_name: str,
        default_value: Any,
        description: Optional[str] = None
    ) -> 'MigrationBuilder':
        """
        Add a property to entities.

        Args:
            entity_type: Entity type to modify
            property_name: Property to add
            default_value: Default value for existing entities
            description: Optional description

        Returns:
            Self for chaining
        """
        step = MigrationStep(
            step_type="add_property",
            description=description or f"Add property '{property_name}' to '{entity_type}'",
            reversible=True,
            metadata={
                "entity_type": entity_type,
                "property_name": property_name,
                "default_value": default_value
            }
        )

        self.steps.append(step)
        return self

    def remove_property(
        self,
        entity_type: str,
        property_name: str,
        description: Optional[str] = None
    ) -> 'MigrationBuilder':
        """
        Remove a property from entities.

        Args:
            entity_type: Entity type to modify
            property_name: Property to remove
            description: Optional description

        Returns:
            Self for chaining
        """
        step = MigrationStep(
            step_type="remove_property",
            description=description or f"Remove property '{property_name}' from '{entity_type}'",
            reversible=True,
            metadata={
                "entity_type": entity_type,
                "property_name": property_name
            }
        )

        self.steps.append(step)
        self.warnings.append(f"Property '{property_name}' will be archived")
        return self

    def rename_property(
        self,
        entity_type: str,
        old_name: str,
        new_name: str,
        description: Optional[str] = None
    ) -> 'MigrationBuilder':
        """
        Rename a property.

        Args:
            entity_type: Entity type to modify
            old_name: Current property name
            new_name: New property name
            description: Optional description

        Returns:
            Self for chaining
        """
        step = MigrationStep(
            step_type="rename_property",
            description=description or f"Rename '{old_name}' to '{new_name}' in '{entity_type}'",
            reversible=True,
            metadata={
                "entity_type": entity_type,
                "old_name": old_name,
                "new_name": new_name
            }
        )

        self.steps.append(step)
        return self

    def transform_values(
        self,
        entity_type: str,
        property_name: str,
        transform_fn: str,
        description: Optional[str] = None
    ) -> 'MigrationBuilder':
        """
        Transform property values.

        Args:
            entity_type: Entity type to modify
            property_name: Property to transform
            transform_fn: Name of transformation function
            description: Optional description

        Returns:
            Self for chaining
        """
        step = MigrationStep(
            step_type="transform_values",
            description=description or f"Transform '{property_name}' in '{entity_type}'",
            reversible=False,  # Value transformations often not reversible
            metadata={
                "entity_type": entity_type,
                "property_name": property_name,
                "transform_fn": transform_fn
            }
        )

        self.steps.append(step)
        self.warnings.append(f"Value transformation is not reversible")
        return self

    def mark_requires_downtime(self) -> 'MigrationBuilder':
        """Mark migration as requiring downtime"""
        self.requires_downtime = True
        self.warnings.append("This migration requires system downtime")
        return self

    def build(self) -> MigrationPlan:
        """
        Build migration plan.

        Returns:
            MigrationPlan
        """
        reversible = all(step.reversible for step in self.steps)

        # Estimate time
        estimated_time = len(self.steps) * 10.0

        plan = MigrationPlan(
            from_version=self.from_version,
            to_version=self.to_version,
            steps=self.steps,
            estimated_time=estimated_time,
            requires_downtime=self.requires_downtime,
            reversible=reversible,
            warnings=self.warnings,
            metadata={
                "custom_plan": True
            }
        )

        return plan


# Convenience functions
async def migrate_data(
    storage: VecnaDBStorageInterface,
    source_ontology: OntologySchema,
    target_ontology: OntologySchema,
    plan: MigrationPlan,
    dry_run: bool = False
) -> MigrationResult:
    """
    Execute data migration.

    Args:
        storage: Storage interface
        source_ontology: Current ontology
        target_ontology: Target ontology
        plan: Migration plan
        dry_run: If True, validate only

    Returns:
        MigrationResult
    """
    executor = MigrationExecutor(storage, source_ontology, target_ontology)
    return await executor.execute_migration(plan, dry_run=dry_run)


def build_migration(
    from_version: str,
    to_version: str
) -> MigrationBuilder:
    """
    Start building a custom migration plan.

    Args:
        from_version: Source version
        to_version: Target version

    Returns:
        MigrationBuilder for fluent API
    """
    return MigrationBuilder(from_version, to_version)
