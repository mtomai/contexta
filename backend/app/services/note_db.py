import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class NoteDB:
    """SQLite database manager for saved notes (pinned AI responses)."""

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
        """Create notes table if it doesn't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    notebook_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_notes_notebook
                ON notes(notebook_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_notes_created
                ON notes(created_at DESC)
            """)

    def create_note(self, notebook_id: str, content: str) -> str:
        """
        Create a new saved note.

        Args:
            notebook_id: Notebook UUID the note belongs to
            content: Note content (pinned AI response)

        Returns:
            Note ID (UUID)
        """
        note_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notes (id, notebook_id, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (note_id, notebook_id, content, now))

        return note_id

    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single note by ID.

        Args:
            note_id: Note UUID

        Returns:
            Note dictionary or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, notebook_id, content, created_at
                FROM notes
                WHERE id = ?
            """, (note_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def list_notes(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        List all notes for a notebook, ordered by most recent first.

        Args:
            notebook_id: Notebook UUID

        Returns:
            List of note dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, notebook_id, content, created_at
                FROM notes
                WHERE notebook_id = ?
                ORDER BY created_at DESC
            """, (notebook_id,))

            return [dict(row) for row in cursor.fetchall()]

    def delete_note(self, note_id: str) -> bool:
        """
        Delete a note by ID.

        Args:
            note_id: Note UUID

        Returns:
            True if deleted, False if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            return cursor.rowcount > 0

    def count_notes(self, notebook_id: str) -> int:
        """
        Count notes for a notebook.

        Args:
            notebook_id: Notebook UUID

        Returns:
            Number of notes
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM notes WHERE notebook_id = ?",
                (notebook_id,)
            )
            return cursor.fetchone()[0]


# Singleton instance
_note_db = None


def get_note_db(db_path: str = "./conversations.db") -> NoteDB:
    """Get or create note DB singleton instance."""
    global _note_db
    if _note_db is None:
        _note_db = NoteDB(db_path)
    return _note_db
