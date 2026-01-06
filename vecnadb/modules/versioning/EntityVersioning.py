"""
VecnaDB Entity Versioning

Provides complete version history tracking for entities.
Every modification creates a new version with full provenance.

Key Features:
1. Immutable version history
2. Time-travel queries (get entity at timestamp)
3. Diff between versions
4. Version rollback
5. Branch tracking (for entity evolution)
6. Change attribution (who/what changed)

Principle: All changes are tracked. Nothing is ever truly deleted.
"""

import time
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
)


class ChangeType(str, Enum):
    """Type of change made to entity"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"  # Soft delete
    RESTORE = "restore"
    MERGE = "merge"
    SPLIT = "split"


class ChangeSource(str, Enum):
    """Source of the change"""
    USER = "user"
    SYSTEM = "system"
    MIGRATION = "migration"
    REASONING = "reasoning"
    RAG = "rag"
    IMPORT = "import"


class PropertyChange(BaseModel):
    """A change to a single property"""
    property_name: str
    old_value: Any
    new_value: Any
    change_type: str  # "added", "modified", "removed"


class EntityVersion(BaseModel):
    """A specific version of an entity"""
    version_id: UUID
    entity_id: UUID
    version_number: int
    entity: KnowledgeEntity
    timestamp: datetime
    change_type: ChangeType
    change_source: ChangeSource
    changed_by: Optional[str] = None  # User ID or system component
    change_description: str
    property_changes: List[PropertyChange] = Field(default_factory=list)
    supersedes: Optional[UUID] = None  # Previous version ID
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VersionDiff(BaseModel):
    """Difference between two versions"""
    from_version: int
    to_version: int
    property_changes: List[PropertyChange]
    embedding_changed: bool
    graph_structure_changed: bool
    change_summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VersionHistory(BaseModel):
    """Complete version history for an entity"""
    entity_id: UUID
    current_version: int
    total_versions: int
    versions: List[EntityVersion]
    created_at: datetime
    last_modified: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_version(self, version_number: int) -> Optional[EntityVersion]:
        """Get specific version by number"""
        return next(
            (v for v in self.versions if v.version_number == version_number),
            None
        )

    def get_version_at_time(self, timestamp: datetime) -> Optional[EntityVersion]:
        """Get version that was current at a specific time"""
        # Find latest version before or at timestamp
        candidates = [
            v for v in self.versions
            if v.timestamp <= timestamp
        ]
        if candidates:
            return max(candidates, key=lambda v: v.timestamp)
        return None


class EntityVersioning:
    """
    Manages entity version history and time-travel queries.

    All entity modifications create new versions. Original versions are
    never modified or deleted, providing complete audit trail.
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        version_storage: Optional[Any] = None  # Separate storage for versions
    ):
        """
        Initialize entity versioning.

        Args:
            storage: VecnaDB storage interface
            version_storage: Optional separate storage for version history
        """
        self.storage = storage
        self.version_storage = version_storage or storage
        self._version_cache: Dict[UUID, VersionHistory] = {}

    async def create_version(
        self,
        entity: KnowledgeEntity,
        change_type: ChangeType,
        change_source: ChangeSource,
        changed_by: Optional[str] = None,
        change_description: Optional[str] = None,
        previous_entity: Optional[KnowledgeEntity] = None
    ) -> EntityVersion:
        """
        Create a new version of an entity.

        Args:
            entity: Entity in its new state
            change_type: Type of change
            change_source: Source of change
            changed_by: Who/what made the change
            change_description: Description of change
            previous_entity: Previous version (for diffs)

        Returns:
            EntityVersion record
        """
        # Get current version number
        history = await self.get_history(entity.id)
        version_number = (history.current_version + 1) if history else 1

        # Calculate property changes
        property_changes = []
        if previous_entity and change_type == ChangeType.UPDATE:
            property_changes = self._calculate_property_changes(
                previous_entity,
                entity
            )

        # Build change description
        if not change_description:
            change_description = self._generate_change_description(
                change_type,
                property_changes
            )

        # Create version record
        version = EntityVersion(
            version_id=uuid4(),
            entity_id=entity.id,
            version_number=version_number,
            entity=entity,
            timestamp=datetime.now(),
            change_type=change_type,
            change_source=change_source,
            changed_by=changed_by,
            change_description=change_description,
            property_changes=property_changes,
            supersedes=entity.supersedes,
            metadata={
                "previous_version": version_number - 1 if version_number > 1 else None
            }
        )

        # Store version
        await self._store_version(version)

        # Update cache
        if entity.id in self._version_cache:
            self._version_cache[entity.id].versions.append(version)
            self._version_cache[entity.id].current_version = version_number
            self._version_cache[entity.id].last_modified = version.timestamp

        return version

    async def get_history(
        self,
        entity_id: UUID,
        max_versions: Optional[int] = None
    ) -> Optional[VersionHistory]:
        """
        Get complete version history for an entity.

        Args:
            entity_id: Entity to get history for
            max_versions: Optional limit on versions returned

        Returns:
            VersionHistory or None if entity not found
        """
        # Check cache
        if entity_id in self._version_cache:
            history = self._version_cache[entity_id]
            if max_versions:
                history.versions = history.versions[-max_versions:]
            return history

        # Load from storage
        versions = await self._load_versions(entity_id)

        if not versions:
            return None

        # Sort by version number
        versions.sort(key=lambda v: v.version_number)

        # Build history
        history = VersionHistory(
            entity_id=entity_id,
            current_version=versions[-1].version_number,
            total_versions=len(versions),
            versions=versions[-max_versions:] if max_versions else versions,
            created_at=versions[0].timestamp,
            last_modified=versions[-1].timestamp,
            metadata={}
        )

        # Cache
        self._version_cache[entity_id] = history

        return history

    async def get_version(
        self,
        entity_id: UUID,
        version_number: int
    ) -> Optional[KnowledgeEntity]:
        """
        Get a specific version of an entity.

        Args:
            entity_id: Entity ID
            version_number: Version number to retrieve

        Returns:
            KnowledgeEntity at that version, or None
        """
        history = await self.get_history(entity_id)

        if not history:
            return None

        version_record = history.get_version(version_number)

        if version_record:
            return version_record.entity

        return None

    async def get_entity_at_time(
        self,
        entity_id: UUID,
        timestamp: datetime
    ) -> Optional[KnowledgeEntity]:
        """
        Time-travel query: get entity as it was at a specific time.

        Args:
            entity_id: Entity ID
            timestamp: Point in time

        Returns:
            KnowledgeEntity as it was at that time, or None
        """
        history = await self.get_history(entity_id)

        if not history:
            return None

        version_record = history.get_version_at_time(timestamp)

        if version_record:
            return version_record.entity

        return None

    async def get_current_version(
        self,
        entity_id: UUID
    ) -> Optional[KnowledgeEntity]:
        """
        Get current (latest) version of entity.

        Args:
            entity_id: Entity ID

        Returns:
            Current KnowledgeEntity or None
        """
        history = await self.get_history(entity_id)

        if not history or not history.versions:
            return None

        return history.versions[-1].entity

    async def diff_versions(
        self,
        entity_id: UUID,
        from_version: int,
        to_version: int
    ) -> Optional[VersionDiff]:
        """
        Calculate diff between two versions.

        Args:
            entity_id: Entity ID
            from_version: Starting version number
            to_version: Ending version number

        Returns:
            VersionDiff or None if versions not found
        """
        history = await self.get_history(entity_id)

        if not history:
            return None

        version_from = history.get_version(from_version)
        version_to = history.get_version(to_version)

        if not version_from or not version_to:
            return None

        # Calculate changes
        property_changes = self._calculate_property_changes(
            version_from.entity,
            version_to.entity
        )

        # Check embedding changes
        embedding_changed = (
            version_from.entity.embeddings != version_to.entity.embeddings
        )

        # Check graph structure changes
        graph_structure_changed = (
            version_from.entity.graph_node_id != version_to.entity.graph_node_id
        )

        # Generate summary
        change_summary = self._generate_diff_summary(
            property_changes,
            embedding_changed,
            graph_structure_changed
        )

        diff = VersionDiff(
            from_version=from_version,
            to_version=to_version,
            property_changes=property_changes,
            embedding_changed=embedding_changed,
            graph_structure_changed=graph_structure_changed,
            change_summary=change_summary,
            metadata={
                "from_timestamp": version_from.timestamp.isoformat(),
                "to_timestamp": version_to.timestamp.isoformat(),
                "time_delta": (version_to.timestamp - version_from.timestamp).total_seconds()
            }
        )

        return diff

    async def rollback_to_version(
        self,
        entity_id: UUID,
        version_number: int,
        changed_by: Optional[str] = None
    ) -> Optional[EntityVersion]:
        """
        Rollback entity to a previous version.

        Creates a new version that is a copy of the specified version.

        Args:
            entity_id: Entity to rollback
            version_number: Version to rollback to
            changed_by: Who initiated the rollback

        Returns:
            New EntityVersion (rollback is a new version)
        """
        # Get target version
        target_entity = await self.get_version(entity_id, version_number)

        if not target_entity:
            return None

        # Get current version for diff
        current_entity = await self.get_current_version(entity_id)

        # Create new version that restores old state
        new_version = await self.create_version(
            entity=target_entity,
            change_type=ChangeType.RESTORE,
            change_source=ChangeSource.USER,
            changed_by=changed_by,
            change_description=f"Rolled back to version {version_number}",
            previous_entity=current_entity
        )

        return new_version

    async def list_changes(
        self,
        entity_id: UUID,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        change_types: Optional[List[ChangeType]] = None
    ) -> List[EntityVersion]:
        """
        List all changes to an entity within a time range.

        Args:
            entity_id: Entity ID
            since: Start time (inclusive)
            until: End time (inclusive)
            change_types: Optional filter by change types

        Returns:
            List of EntityVersion records
        """
        history = await self.get_history(entity_id)

        if not history:
            return []

        # Filter by time range
        filtered = history.versions

        if since:
            filtered = [v for v in filtered if v.timestamp >= since]

        if until:
            filtered = [v for v in filtered if v.timestamp <= until]

        # Filter by change type
        if change_types:
            filtered = [v for v in filtered if v.change_type in change_types]

        return filtered

    def _calculate_property_changes(
        self,
        old_entity: KnowledgeEntity,
        new_entity: KnowledgeEntity
    ) -> List[PropertyChange]:
        """Calculate property changes between two entities"""
        changes = []

        old_props = old_entity.properties
        new_props = new_entity.properties

        # Find all property keys
        all_keys = set(old_props.keys()).union(set(new_props.keys()))

        for key in all_keys:
            old_value = old_props.get(key)
            new_value = new_props.get(key)

            if old_value is None and new_value is not None:
                # Property added
                changes.append(PropertyChange(
                    property_name=key,
                    old_value=None,
                    new_value=new_value,
                    change_type="added"
                ))

            elif old_value is not None and new_value is None:
                # Property removed
                changes.append(PropertyChange(
                    property_name=key,
                    old_value=old_value,
                    new_value=None,
                    change_type="removed"
                ))

            elif old_value != new_value:
                # Property modified
                changes.append(PropertyChange(
                    property_name=key,
                    old_value=old_value,
                    new_value=new_value,
                    change_type="modified"
                ))

        return changes

    def _generate_change_description(
        self,
        change_type: ChangeType,
        property_changes: List[PropertyChange]
    ) -> str:
        """Generate human-readable change description"""
        if change_type == ChangeType.CREATE:
            return "Entity created"

        elif change_type == ChangeType.DELETE:
            return "Entity deleted"

        elif change_type == ChangeType.UPDATE:
            if not property_changes:
                return "Entity updated (no property changes)"

            change_count = len(property_changes)
            changed_props = [c.property_name for c in property_changes[:3]]
            props_str = ", ".join(changed_props)

            if change_count > 3:
                return f"Updated {change_count} properties: {props_str}, ..."
            else:
                return f"Updated properties: {props_str}"

        else:
            return f"Entity {change_type.value}"

    def _generate_diff_summary(
        self,
        property_changes: List[PropertyChange],
        embedding_changed: bool,
        graph_structure_changed: bool
    ) -> str:
        """Generate summary of diff"""
        parts = []

        if property_changes:
            parts.append(f"{len(property_changes)} property changes")

        if embedding_changed:
            parts.append("embeddings changed")

        if graph_structure_changed:
            parts.append("graph structure changed")

        if not parts:
            return "No changes"

        return "; ".join(parts)

    async def _store_version(self, version: EntityVersion):
        """Store version record (implementation depends on storage backend)"""
        # This is a placeholder - actual implementation would store to database
        # For now, we rely on in-memory cache
        pass

    async def _load_versions(self, entity_id: UUID) -> List[EntityVersion]:
        """Load version history from storage (placeholder)"""
        # This is a placeholder - actual implementation would load from database
        # For now, return empty list
        return []

    async def get_entity_timeline(
        self,
        entity_id: UUID,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Get a visual timeline of entity changes.

        Args:
            entity_id: Entity ID
            include_metadata: Include detailed metadata

        Returns:
            Timeline data structure
        """
        history = await self.get_history(entity_id)

        if not history:
            return {"error": "Entity not found"}

        timeline = {
            "entity_id": str(entity_id),
            "total_versions": history.total_versions,
            "created_at": history.created_at.isoformat(),
            "last_modified": history.last_modified.isoformat(),
            "events": []
        }

        for version in history.versions:
            event = {
                "version": version.version_number,
                "timestamp": version.timestamp.isoformat(),
                "change_type": version.change_type,
                "change_source": version.change_source,
                "description": version.change_description,
                "changed_by": version.changed_by
            }

            if include_metadata:
                event["property_changes"] = len(version.property_changes)
                event["properties_changed"] = [
                    c.property_name for c in version.property_changes
                ]

            timeline["events"].append(event)

        return timeline


# Convenience functions
async def track_entity_change(
    entity: KnowledgeEntity,
    change_type: ChangeType,
    storage: VecnaDBStorageInterface,
    changed_by: Optional[str] = None,
    change_source: ChangeSource = ChangeSource.USER,
    previous_entity: Optional[KnowledgeEntity] = None
) -> EntityVersion:
    """
    Track a change to an entity.

    Args:
        entity: Entity in new state
        change_type: Type of change
        storage: Storage interface
        changed_by: Who made the change
        change_source: Source of change
        previous_entity: Previous version

    Returns:
        EntityVersion record
    """
    versioning = EntityVersioning(storage)
    return await versioning.create_version(
        entity=entity,
        change_type=change_type,
        change_source=change_source,
        changed_by=changed_by,
        previous_entity=previous_entity
    )


async def time_travel(
    entity_id: UUID,
    timestamp: datetime,
    storage: VecnaDBStorageInterface
) -> Optional[KnowledgeEntity]:
    """
    Get entity as it was at a specific time.

    Args:
        entity_id: Entity ID
        timestamp: Point in time
        storage: Storage interface

    Returns:
        Entity at that time or None
    """
    versioning = EntityVersioning(storage)
    return await versioning.get_entity_at_time(entity_id, timestamp)


async def compare_versions(
    entity_id: UUID,
    version1: int,
    version2: int,
    storage: VecnaDBStorageInterface
) -> Optional[VersionDiff]:
    """
    Compare two versions of an entity.

    Args:
        entity_id: Entity ID
        version1: First version
        version2: Second version
        storage: Storage interface

    Returns:
        VersionDiff or None
    """
    versioning = EntityVersioning(storage)
    return await versioning.diff_versions(entity_id, version1, version2)
