"""
Storage migration utility for Zen MCP Server knowledge base.

Handles migration of context knowledge base from temporary to persistent storage.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def migrate_context_storage(
    old_path: str = "/tmp/zen-context-kb", new_path: str = None, force: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Migrate context knowledge base from old location to new location.

    Args:
        old_path: Source path (default: /tmp/zen-context-kb)
        new_path: Destination path (default: ./data/kb)
        force: Force migration even if destination exists

    Returns:
        Tuple of (success, error_message)
    """
    if new_path is None:
        new_path = os.environ.get("STORAGE_DIR", "./data/kb")

    old_dir = Path(old_path)
    new_dir = Path(new_path)

    # Check if migration is needed
    if not old_dir.exists():
        logger.info(f"No existing knowledge base found at {old_path}, nothing to migrate")
        return True, None

    # Check if destination exists
    if new_dir.exists() and not force:
        # Check if it's empty or has content
        if any(new_dir.iterdir()):
            logger.warning(
                f"Destination {new_path} already exists and contains data. "
                "Use force=True to overwrite or manually merge the data."
            )
            return False, f"Destination {new_path} already contains data"

    try:
        # Create parent directory if needed
        new_dir.parent.mkdir(parents=True, exist_ok=True)

        # Count items to migrate
        projects_dir = old_dir / "projects"
        project_count = 0
        entry_count = 0

        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    project_count += 1
                    entries_dir = project_dir / "entries"
                    if entries_dir.exists():
                        entry_count += len(list(entries_dir.glob("*.json")))

        logger.info(
            f"Starting migration: {project_count} projects, {entry_count} entries " f"from {old_path} to {new_path}"
        )

        # Perform the migration
        if new_dir.exists() and force:
            logger.warning(f"Force flag set, removing existing directory at {new_path}")
            shutil.rmtree(new_dir)

        # Move the entire directory
        shutil.move(str(old_dir), str(new_dir))

        logger.info(
            f"Successfully migrated knowledge base from {old_path} to {new_path}. "
            f"Migrated {project_count} projects with {entry_count} total entries."
        )

        # Create a migration marker file
        marker_file = new_dir / ".migrated_from_tmp"
        with open(marker_file, "w") as f:
            migration_info = {
                "migrated_from": old_path,
                "migrated_to": new_path,
                "migration_date": str(Path(old_path).stat().st_mtime if old_dir.exists() else "unknown"),
                "project_count": project_count,
                "entry_count": entry_count,
            }
            json.dump(migration_info, f, indent=2)

        return True, None

    except PermissionError as e:
        error_msg = f"Permission denied during migration: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Migration failed: {e}"
        logger.error(error_msg)
        return False, error_msg


def check_migration_needed(old_path: str = "/tmp/zen-context-kb", new_path: str = None) -> bool:
    """
    Check if migration is needed.

    Returns True if old path exists and new path doesn't, False otherwise.
    """
    if new_path is None:
        new_path = os.environ.get("STORAGE_DIR", "./data/kb")

    old_dir = Path(old_path)
    new_dir = Path(new_path)

    # Migration needed if old exists but new doesn't
    return old_dir.exists() and not new_dir.exists()


def ensure_storage_directory(path: str = None) -> bool:
    """
    Ensure the storage directory exists with proper permissions.

    Args:
        path: Directory path to ensure exists

    Returns:
        True if directory exists or was created, False on error
    """
    if path is None:
        path = os.environ.get("STORAGE_DIR", "./data/kb")

    try:
        dir_path = Path(path)

        # Check if parent exists first
        if not dir_path.parent.exists():
            logger.warning(f"Parent directory {dir_path.parent} does not exist, attempting to create full path")

        dir_path.mkdir(parents=True, exist_ok=True)

        # Ensure it's writable
        test_file = dir_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
            logger.debug(f"Storage directory {path} is ready and writable")
            return True
        except PermissionError:
            # Try to fix permissions if we own the directory
            try:
                os.chmod(str(dir_path), 0o755)
                test_file.touch()
                test_file.unlink()
                logger.info(f"Fixed permissions for {path}")
                return True
            except Exception:
                logger.error(f"Permission denied: Cannot write to {path} even after chmod attempt")
                return False

    except PermissionError:
        logger.error(f"Permission denied: Cannot create directory {path}")
        return False
    except Exception as e:
        logger.error(f"Failed to ensure storage directory: {e}")
        return False


if __name__ == "__main__":
    # Test migration when run directly
    logging.basicConfig(level=logging.INFO)
    success, error = migrate_context_storage()
    if success:
        print("Migration completed successfully")
    else:
        print(f"Migration failed: {error}")
