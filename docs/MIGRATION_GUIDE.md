# VecnaDB Migration Guide

Guide for migrating between VecnaDB ontology versions and upgrading the system.

---

## Ontology Migrations

### Planning a Migration

```python
from vecnadb.modules.versioning import OntologyEvolution

evolution = OntologyEvolution()

# Generate migration plan
plan = await evolution.generate_migration_plan(
    ontology_id=ontology.id,
    from_version="1.0.0",
    to_version="2.0.0"
)

print(f"Migration steps: {len(plan.steps)}")
print(f"Estimated time: {plan.estimated_time}s")
print(f"Requires downtime: {plan.requires_downtime}")
print(f"Reversible: {plan.reversible}")

for warning in plan.warnings:
    print(f"WARNING: {warning}")
```

### Executing a Migration

```python
from vecnadb.modules.versioning import MigrationExecutor, MigrationStatus

executor = MigrationExecutor(storage, old_ontology, new_ontology)

# 1. DRY RUN FIRST
dry_result = await executor.execute_migration(plan, dry_run=True)

if dry_result.status == MigrationStatus.COMPLETED:
    print("✅ Dry run successful")

    # 2. Execute for real
    result = await executor.execute_migration(
        plan=plan,
        dry_run=False,
        batch_size=100
    )

    if result.status == MigrationStatus.COMPLETED:
        print(f"✅ Migrated {result.entities_migrated} entities")
    else:
        print(f"❌ Migration failed: {result.errors}")
        # Rollback
        await executor.rollback_migration(result.migration_id)
```

### Custom Migrations

```python
from vecnadb.modules.versioning import build_migration

migration = (
    build_migration("1.0.0", "2.0.0")
    .add_property("Person", "email", default_value="")
    .rename_property("Person", "full_name", "name")
    .remove_property("Document", "deprecated_field")
    .build()
)

result = await executor.execute_migration(migration)
```

---

## Breaking vs. Non-Breaking Changes

### Non-Breaking (MINOR/PATCH)
- Add optional property
- Add new entity type
- Add new relation type
- Relax constraint

### Breaking (MAJOR)
- Remove property
- Remove entity type
- Make property required
- Tighten constraint

---

## Version Compatibility

### Check Compatibility

```python
compatibility, changes = await evolution.check_compatibility(
    old_ontology,
    new_ontology
)

if compatibility == CompatibilityLevel.BREAKING:
    print("⚠️ BREAKING CHANGES - Requires major version bump")
elif compatibility == CompatibilityLevel.COMPATIBLE_WITH_MIGRATION:
    print("✓ Compatible with migration - Minor version bump")
else:
    print("✓ Fully compatible - Patch version bump")
```

---

## Best Practices

1. **Always dry run first**
2. **Backup data before migration**
3. **Test on staging environment**
4. **Monitor migration progress**
5. **Keep rollback plan ready**

See `SPRINT_7_SUMMARY.md` for detailed versioning documentation.
