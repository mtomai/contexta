import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class CustomPromptsDB:
    """SQLite database manager for custom prompts/functions."""

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
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
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
        """Create tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS custom_prompts (
                    id TEXT PRIMARY KEY,
                    notebook_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    prompt_template TEXT NOT NULL,
                    icon TEXT DEFAULT 'Sparkles',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_custom_prompts_notebook
                ON custom_prompts(notebook_id)
            """)

    def create_custom_prompt(
        self,
        notebook_id: str,
        name: str,
        prompt_template: str,
        icon: str = "Sparkles"
    ) -> str:
        """
        Create a new custom prompt.

        Args:
            notebook_id: Notebook UUID
            name: Display name (e.g., "Riassunto")
            prompt_template: The prompt text to send to LLM
            icon: Icon name (default "Sparkles")

        Returns:
            prompt_id: UUID of created prompt
        """
        prompt_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO custom_prompts (id, notebook_id, name, prompt_template, icon, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (prompt_id, notebook_id, name, prompt_template, icon, now, now))

        return prompt_id

    def get_custom_prompts(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        Get all custom prompts for a notebook.

        Args:
            notebook_id: Notebook UUID

        Returns:
            List of custom prompt dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, notebook_id, name, prompt_template, icon, created_at, updated_at
                FROM custom_prompts
                WHERE notebook_id = ?
                ORDER BY created_at ASC
            """, (notebook_id,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_custom_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single custom prompt by ID.

        Args:
            prompt_id: Prompt UUID

        Returns:
            Custom prompt dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, notebook_id, name, prompt_template, icon, created_at, updated_at
                FROM custom_prompts
                WHERE id = ?
            """, (prompt_id,))

            row = cursor.fetchone()
            return dict(row) if row else None

    def update_custom_prompt(
        self,
        prompt_id: str,
        name: Optional[str] = None,
        prompt_template: Optional[str] = None,
        icon: Optional[str] = None
    ) -> bool:
        """
        Update a custom prompt.

        Args:
            prompt_id: Prompt UUID
            name: New name (optional)
            prompt_template: New template (optional)
            icon: New icon (optional)

        Returns:
            True if updated, False if not found
        """
        updates = []
        values = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if prompt_template is not None:
            updates.append("prompt_template = ?")
            values.append(prompt_template)
        if icon is not None:
            updates.append("icon = ?")
            values.append(icon)

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(prompt_id)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE custom_prompts
                SET {', '.join(updates)}
                WHERE id = ?
            """, values)

            return cursor.rowcount > 0

    def delete_custom_prompt(self, prompt_id: str) -> bool:
        """
        Delete a custom prompt.

        Args:
            prompt_id: Prompt UUID

        Returns:
            True if deleted, False if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM custom_prompts WHERE id = ?", (prompt_id,))
            return cursor.rowcount > 0


# Singleton instance
_custom_prompts_db_instance = None


def get_custom_prompts_db(db_path: str = "./conversations.db") -> CustomPromptsDB:
    """
    Get singleton instance of CustomPromptsDB.

    Args:
        db_path: Path to SQLite database file

    Returns:
        CustomPromptsDB instance
    """
    global _custom_prompts_db_instance
    if _custom_prompts_db_instance is None:
        _custom_prompts_db_instance = CustomPromptsDB(db_path)
    return _custom_prompts_db_instance
