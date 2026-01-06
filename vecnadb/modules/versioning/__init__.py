"""
VecnaDB Versioning & Audit Module

Provides complete version history, schema evolution, data migration, and audit logging.

Key Components:
1. EntityVersioning: Track all entity changes with time-travel queries
2. OntologyEvolution: Schema versioning with semantic versioning
3. MigrationTools: Execute migrations between ontology versions
4. AuditLogger: Complete audit trail for all operations

Features:
- Immutable version history
- Time-travel queries
- Semantic versioning for ontologies
- Automatic migration planning
- Breaking change detection
- Rollback support
- Complete audit trail
- Compliance reporting

Principles:
- All changes are tracked (nothing is ever truly deleted)
- Migrations are traceable and reversible
- All operations are auditable
"""

from vecnadb.modules.versioning.EntityVersioning import (
    EntityVersioning,
    EntityVersion,
    VersionHistory,
    VersionDiff,
    PropertyChange,
    ChangeType,
    ChangeSource,
    track_entity_change,
    time_travel,
    compare_versions,
)

from vecnadb.modules.versioning.OntologyEvolution import (
    OntologyEvolution,
    OntologyVersion,
    SchemaChange,
    SchemaChangeType,
    CompatibilityLevel,
    MigrationPlan,
    MigrationStep,
    evolve_ontology,
    plan_migration,
)

from vecnadb.modules.versioning.MigrationTools import (
    MigrationExecutor,
    MigrationBuilder,
    MigrationResult,
    MigrationStatus,
    MigrationProgress,
    migrate_data,
    build_migration,
)

from vecnadb.modules.versioning.AuditLogger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditActor,
    AuditQuery,
    AuditReport,
    audit,
    get_audit_logger,
    set_audit_logger,
)

__all__ = [
    # Entity Versioning
    "EntityVersioning",
    "EntityVersion",
    "VersionHistory",
    "VersionDiff",
    "PropertyChange",
    "ChangeType",
    "ChangeSource",
    "track_entity_change",
    "time_travel",
    "compare_versions",
    # Ontology Evolution
    "OntologyEvolution",
    "OntologyVersion",
    "SchemaChange",
    "SchemaChangeType",
    "CompatibilityLevel",
    "MigrationPlan",
    "MigrationStep",
    "evolve_ontology",
    "plan_migration",
    # Migration Tools
    "MigrationExecutor",
    "MigrationBuilder",
    "MigrationResult",
    "MigrationStatus",
    "MigrationProgress",
    "migrate_data",
    "build_migration",
    # Audit Logging
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "AuditActor",
    "AuditQuery",
    "AuditReport",
    "audit",
    "get_audit_logger",
    "set_audit_logger",
]
