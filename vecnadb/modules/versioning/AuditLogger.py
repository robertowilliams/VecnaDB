"""
VecnaDB Audit Logger

Complete audit trail for all operations on the knowledge graph.

Key Features:
1. Log all CRUD operations
2. Track user/system actions
3. Record access patterns
4. Security audit trail
5. Compliance reporting
6. Query audit logs

Principle: All operations must be auditable and traceable.
"""

import time
from typing import List, Optional, Dict, Any, Set
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum
import json

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    """Type of audited event"""
    # Entity operations
    ENTITY_CREATE = "entity_create"
    ENTITY_READ = "entity_read"
    ENTITY_UPDATE = "entity_update"
    ENTITY_DELETE = "entity_delete"

    # Relation operations
    RELATION_CREATE = "relation_create"
    RELATION_READ = "relation_read"
    RELATION_UPDATE = "relation_update"
    RELATION_DELETE = "relation_delete"

    # Query operations
    QUERY_EXECUTE = "query_execute"
    SEARCH_EXECUTE = "search_execute"

    # RAG operations
    RAG_QUERY = "rag_query"
    RAG_ANSWER_GENERATE = "rag_answer_generate"

    # Reasoning operations
    REASONING_EXECUTE = "reasoning_execute"
    INFERENCE_CREATE = "inference_create"

    # Schema operations
    ONTOLOGY_CREATE = "ontology_create"
    ONTOLOGY_UPDATE = "ontology_update"
    ONTOLOGY_MIGRATE = "ontology_migrate"

    # Migration operations
    MIGRATION_START = "migration_start"
    MIGRATION_COMPLETE = "migration_complete"
    MIGRATION_ROLLBACK = "migration_rollback"

    # Security events
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"


class AuditSeverity(str, Enum):
    """Severity level of audit event"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditActor(BaseModel):
    """Actor who performed the action"""
    actor_type: str  # "user", "system", "service"
    actor_id: Optional[str] = None  # User ID, system component, etc.
    actor_name: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditEvent(BaseModel):
    """A single audit log entry"""
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO

    # Who
    actor: AuditActor

    # What
    resource_type: str  # "entity", "relation", "ontology", etc.
    resource_id: Optional[UUID] = None
    action: str  # "create", "read", "update", "delete", etc.

    # Context
    description: str
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None

    # Result
    success: bool = True
    error_message: Optional[str] = None

    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class AuditQuery(BaseModel):
    """Query for searching audit logs"""
    event_types: Optional[List[AuditEventType]] = None
    actor_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[UUID] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    severity: Optional[AuditSeverity] = None
    success_only: Optional[bool] = None
    tags: Optional[List[str]] = None
    limit: int = 100


class AuditReport(BaseModel):
    """Audit report for a time period"""
    report_id: UUID = Field(default_factory=uuid4)
    generated_at: datetime = Field(default_factory=datetime.now)
    start_time: datetime
    end_time: datetime

    # Summary statistics
    total_events: int
    events_by_type: Dict[AuditEventType, int]
    events_by_actor: Dict[str, int]
    events_by_severity: Dict[AuditSeverity, int]

    # Security metrics
    failed_operations: int
    access_denied_count: int
    auth_failures: int

    # Top actors
    top_actors: List[Tuple[str, int]]

    # Most modified resources
    top_resources: List[Tuple[str, int]]

    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """
    Comprehensive audit logging for VecnaDB operations.

    Logs all operations for security, compliance, and debugging.
    """

    def __init__(
        self,
        storage: Optional[Any] = None,
        buffer_size: int = 1000
    ):
        """
        Initialize audit logger.

        Args:
            storage: Storage backend for audit logs
            buffer_size: Size of in-memory buffer before flush
        """
        self.storage = storage
        self.buffer_size = buffer_size
        self._buffer: List[AuditEvent] = []
        self._total_events = 0

    async def log_event(
        self,
        event_type: AuditEventType,
        actor: AuditActor,
        resource_type: str,
        action: str,
        description: str,
        resource_id: Optional[UUID] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        success: bool = True,
        error_message: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            actor: Who performed the action
            resource_type: Type of resource
            action: Action performed
            description: Human-readable description
            resource_id: Optional resource ID
            severity: Event severity
            success: Whether operation succeeded
            error_message: Optional error message
            before_state: State before operation
            after_state: State after operation
            metadata: Additional metadata
            tags: Event tags for filtering

        Returns:
            AuditEvent record
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            actor=actor,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            description=description,
            before_state=before_state,
            after_state=after_state,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
            tags=tags or []
        )

        # Add to buffer
        self._buffer.append(event)
        self._total_events += 1

        # Flush if buffer full
        if len(self._buffer) >= self.buffer_size:
            await self.flush()

        return event

    async def log_entity_operation(
        self,
        operation: str,  # "create", "read", "update", "delete"
        entity_id: UUID,
        actor: AuditActor,
        success: bool = True,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> AuditEvent:
        """Log an entity operation"""
        event_type_map = {
            "create": AuditEventType.ENTITY_CREATE,
            "read": AuditEventType.ENTITY_READ,
            "update": AuditEventType.ENTITY_UPDATE,
            "delete": AuditEventType.ENTITY_DELETE
        }

        return await self.log_event(
            event_type=event_type_map.get(operation, AuditEventType.ENTITY_UPDATE),
            actor=actor,
            resource_type="entity",
            resource_id=entity_id,
            action=operation,
            description=f"Entity {operation}: {entity_id}",
            success=success,
            before_state=before_state,
            after_state=after_state,
            error_message=error_message
        )

    async def log_query(
        self,
        query_text: str,
        actor: AuditActor,
        results_count: int,
        execution_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditEvent:
        """Log a query execution"""
        return await self.log_event(
            event_type=AuditEventType.QUERY_EXECUTE,
            actor=actor,
            resource_type="query",
            action="execute",
            description=f"Query executed: {query_text[:100]}...",
            success=success,
            error_message=error_message,
            metadata={
                "query_text": query_text,
                "results_count": results_count,
                "execution_time_ms": execution_time_ms
            },
            tags=["query"]
        )

    async def log_rag_operation(
        self,
        query: str,
        answer: str,
        actor: AuditActor,
        context_entities: List[UUID],
        confidence: float,
        hallucination_risk: float
    ) -> AuditEvent:
        """Log a RAG answer generation"""
        return await self.log_event(
            event_type=AuditEventType.RAG_ANSWER_GENERATE,
            actor=actor,
            resource_type="rag_answer",
            action="generate",
            description=f"RAG answer generated for: {query[:100]}...",
            success=True,
            metadata={
                "query": query,
                "answer_length": len(answer),
                "context_entities": [str(e) for e in context_entities],
                "confidence": confidence,
                "hallucination_risk": hallucination_risk
            },
            tags=["rag", "llm"]
        )

    async def log_migration(
        self,
        migration_id: UUID,
        from_version: str,
        to_version: str,
        actor: AuditActor,
        status: str,
        entities_migrated: int,
        duration_seconds: float,
        success: bool = True,
        errors: Optional[List[str]] = None
    ) -> AuditEvent:
        """Log a migration operation"""
        event_type = (
            AuditEventType.MIGRATION_COMPLETE if status == "completed"
            else AuditEventType.MIGRATION_ROLLBACK if status == "rolled_back"
            else AuditEventType.MIGRATION_START
        )

        return await self.log_event(
            event_type=event_type,
            actor=actor,
            resource_type="migration",
            resource_id=migration_id,
            action=status,
            description=f"Migration {from_version} → {to_version}: {status}",
            success=success,
            error_message="; ".join(errors) if errors else None,
            metadata={
                "from_version": from_version,
                "to_version": to_version,
                "entities_migrated": entities_migrated,
                "duration_seconds": duration_seconds
            },
            tags=["migration", "schema"]
        )

    async def log_security_event(
        self,
        event_type: AuditEventType,
        actor: AuditActor,
        description: str,
        success: bool,
        resource_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log a security-related event"""
        severity = (
            AuditSeverity.CRITICAL if not success and event_type == AuditEventType.AUTH_FAILURE
            else AuditSeverity.WARNING if not success
            else AuditSeverity.INFO
        )

        return await self.log_event(
            event_type=event_type,
            actor=actor,
            resource_type="security",
            resource_id=resource_id,
            action=event_type.value,
            description=description,
            severity=severity,
            success=success,
            metadata=metadata or {},
            tags=["security"]
        )

    async def query_logs(
        self,
        query: AuditQuery
    ) -> List[AuditEvent]:
        """
        Query audit logs.

        Args:
            query: Audit query parameters

        Returns:
            List of matching audit events
        """
        # Filter buffer (in-memory search for now)
        results = self._buffer.copy()

        # Apply filters
        if query.event_types:
            results = [e for e in results if e.event_type in query.event_types]

        if query.actor_id:
            results = [e for e in results if e.actor.actor_id == query.actor_id]

        if query.resource_type:
            results = [e for e in results if e.resource_type == query.resource_type]

        if query.resource_id:
            results = [e for e in results if e.resource_id == query.resource_id]

        if query.start_time:
            results = [e for e in results if e.timestamp >= query.start_time]

        if query.end_time:
            results = [e for e in results if e.timestamp <= query.end_time]

        if query.severity:
            results = [e for e in results if e.severity == query.severity]

        if query.success_only is not None:
            results = [e for e in results if e.success == query.success_only]

        if query.tags:
            results = [
                e for e in results
                if any(tag in e.tags for tag in query.tags)
            ]

        # Sort by timestamp (newest first)
        results.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        return results[:query.limit]

    async def generate_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> AuditReport:
        """
        Generate audit report for a time period.

        Args:
            start_time: Report start time
            end_time: Report end time

        Returns:
            AuditReport with statistics
        """
        # Query events in time range
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            limit=1000000  # Get all
        )
        events = await self.query_logs(query)

        # Calculate statistics
        events_by_type: Dict[AuditEventType, int] = {}
        events_by_actor: Dict[str, int] = {}
        events_by_severity: Dict[AuditSeverity, int] = {}

        failed_operations = 0
        access_denied_count = 0
        auth_failures = 0

        for event in events:
            # By type
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1

            # By actor
            actor_key = event.actor.actor_id or "unknown"
            events_by_actor[actor_key] = events_by_actor.get(actor_key, 0) + 1

            # By severity
            events_by_severity[event.severity] = events_by_severity.get(event.severity, 0) + 1

            # Security metrics
            if not event.success:
                failed_operations += 1

            if event.event_type == AuditEventType.ACCESS_DENIED:
                access_denied_count += 1

            if event.event_type == AuditEventType.AUTH_FAILURE:
                auth_failures += 1

        # Top actors
        top_actors = sorted(
            events_by_actor.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Most modified resources (simplified)
        top_resources = []

        report = AuditReport(
            start_time=start_time,
            end_time=end_time,
            total_events=len(events),
            events_by_type=events_by_type,
            events_by_actor=events_by_actor,
            events_by_severity=events_by_severity,
            failed_operations=failed_operations,
            access_denied_count=access_denied_count,
            auth_failures=auth_failures,
            top_actors=top_actors,
            top_resources=top_resources,
            metadata={
                "duration_hours": (end_time - start_time).total_seconds() / 3600
            }
        )

        return report

    async def flush(self):
        """Flush buffer to storage"""
        if not self._buffer:
            return

        if self.storage:
            # Store events to persistent storage
            await self._persist_events(self._buffer)

        # Clear buffer
        self._buffer.clear()

    async def _persist_events(self, events: List[AuditEvent]):
        """Persist events to storage (placeholder)"""
        # Actual implementation would write to database/file
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get current audit logger statistics"""
        return {
            "total_events_logged": self._total_events,
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.buffer_size
        }


# Convenience functions
async def audit(
    event_type: AuditEventType,
    actor_id: str,
    resource_type: str,
    action: str,
    description: str,
    logger: Optional[AuditLogger] = None,
    **kwargs
) -> AuditEvent:
    """
    Quick audit logging.

    Args:
        event_type: Event type
        actor_id: Actor ID
        resource_type: Resource type
        action: Action performed
        description: Description
        logger: Optional logger instance
        **kwargs: Additional audit event parameters

    Returns:
        AuditEvent
    """
    if not logger:
        logger = AuditLogger()

    actor = AuditActor(
        actor_type="user",
        actor_id=actor_id
    )

    return await logger.log_event(
        event_type=event_type,
        actor=actor,
        resource_type=resource_type,
        action=action,
        description=description,
        **kwargs
    )


# Global logger instance (singleton pattern)
_global_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AuditLogger()
    return _global_logger


def set_audit_logger(logger: AuditLogger):
    """Set global audit logger instance"""
    global _global_logger
    _global_logger = logger
