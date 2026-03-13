import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class NotebookDB:
    """SQLite database manager for notebooks."""

    def __init__(self, db_path: str = "./conversations.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic commit/rollback."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dictionary-like access
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_db(self):
        """Create notebooks table if it doesn't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create notebooks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notebooks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)

            # Create index on updated_at for sorting
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_notebooks_updated
                ON notebooks(updated_at DESC)
            """)

    def create_notebook(
        self,
        name: str,
        description: Optional[str] = None
    ) -> str:
        """
        Create a new notebook.

        Args:
            name: Notebook name
            description: Optional description

        Returns:
            Notebook ID (UUID)
        """
        notebook_id = str(uuid.uuid4())
        now = datetime.utcnow()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notebooks (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (notebook_id, name, description, now, now))

        return notebook_id

    def list_notebooks(self) -> List[Dict[str, Any]]:
        """
        List all notebooks ordered by most recently updated.

        Returns:
            List of notebook dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, created_at, updated_at
                FROM notebooks
                ORDER BY updated_at DESC
            """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_notebook(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """
        Get notebook details by ID.

        Args:
            notebook_id: Notebook UUID

        Returns:
            Notebook dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, created_at, updated_at
                FROM notebooks
                WHERE id = ?
            """, (notebook_id,))

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_notebook_with_stats(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """
        Get notebook details with statistics (document count, conversation count).

        Args:
            notebook_id: Notebook UUID

        Returns:
            Notebook dictionary with stats or None if not found
        """
        notebook = self.get_notebook(notebook_id)
        if not notebook:
            return None

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Count conversations
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM conversations
                WHERE notebook_id = ?
            """, (notebook_id,))
            conversation_count = cursor.fetchone()["count"]

            # Add stats to notebook
            notebook["conversation_count"] = conversation_count
            notebook["document_count"] = 0  # Will be filled by vector_store

        return notebook

    def update_notebook(
        self,
        notebook_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Update notebook name and/or description.

        Args:
            notebook_id: Notebook UUID
            name: New name (optional)
            description: New description (optional)

        Returns:
            True if updated, False if notebook not found
        """
        # Check if notebook exists
        if not self.get_notebook(notebook_id):
            return False

        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if not updates:
            return True  # Nothing to update

        # Always update updated_at
        updates.append("updated_at = ?")
        params.append(datetime.utcnow())
        params.append(notebook_id)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = f"""
                UPDATE notebooks
                SET {', '.join(updates)}
                WHERE id = ?
            """
            cursor.execute(query, params)

        return True

    def delete_notebook(self, notebook_id: str) -> bool:
        """
        Delete notebook. Conversations will be deleted due to CASCADE.

        Args:
            notebook_id: Notebook UUID

        Returns:
            True if deleted, False if notebook not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
            return cursor.rowcount > 0

    def notebook_exists(self, notebook_id: str) -> bool:
        """
        Check if a notebook exists.

        Args:
            notebook_id: Notebook UUID

        Returns:
            True if exists, False otherwise
        """
        return self.get_notebook(notebook_id) is not None


# Singleton instance
_notebook_db = None


def get_notebook_db(db_path: str = "./conversations.db") -> NotebookDB:
    """Get or create notebook DB singleton instance."""
    global _notebook_db
    if _notebook_db is None:
        _notebook_db = NotebookDB(db_path)
    return _notebook_db
