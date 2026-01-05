#!/usr/bin/env python3
"""
Script to update all 'cognee' imports to 'vecnadb' throughout the codebase.
"""

import os
import re
from pathlib import Path


def update_imports_in_file(file_path: Path) -> tuple[int, bool]:
    """
    Update all cognee imports to vecnadb in a single file.

    Returns:
        Tuple of (number of replacements, whether file was modified)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, False

    original_content = content
    replacements = 0

    # Replace import statements
    patterns = [
        (r'\bfrom cognee\.', 'from vecnadb.'),
        (r'\bfrom cognee\b', 'from vecnadb'),
        (r'\bimport cognee\.', 'import vecnadb.'),
        (r'\bimport cognee\b', 'import vecnadb'),
        (r'\bcognee\.', 'vecnadb.'),
    ]

    for pattern, replacement in patterns:
        new_content, count = re.subn(pattern, replacement, content)
        replacements += count
        content = new_content

    # Only write if changed
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return replacements, True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return 0, False

    return 0, False


def main():
    """Main function to update all Python files."""
    root_dir = Path("/Users/roberto/Documents/VecnaDB")

    # Directories to process
    dirs_to_process = [
        root_dir / "vecnadb",
        root_dir / "distributed",
        root_dir / "alembic",
    ]

    # Files to skip
    skip_patterns = [
        ".git",
        "__pycache__",
        ".pytest_cache",
        "venv",
        "env",
        ".data",
        "node_modules",
        "cognee-frontend",
        "cognee-mcp",
    ]

    total_files = 0
    total_modified = 0
    total_replacements = 0

    for dir_path in dirs_to_process:
        if not dir_path.exists():
            continue

        print(f"\nProcessing directory: {dir_path}")

        for file_path in dir_path.rglob("*.py"):
            # Skip files in excluded directories
            if any(skip in str(file_path) for skip in skip_patterns):
                continue

            total_files += 1
            replacements, modified = update_imports_in_file(file_path)

            if modified:
                total_modified += 1
                total_replacements += replacements
                print(f"  ✓ {file_path.relative_to(root_dir)} ({replacements} replacements)")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files scanned: {total_files}")
    print(f"  Files modified: {total_modified}")
    print(f"  Total replacements: {total_replacements}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
