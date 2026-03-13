import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from pathlib import Path


class ConversationDB:
    """SQLite database manager for conversations and messages."""

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
        """Create tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    message_count INTEGER DEFAULT 0
                )
            """)

            # Create index on updated_at for sorting
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC)
            """)

            # Migration: Add notebook_id column if it doesn't exist
            cursor.execute("PRAGMA table_info(conversations)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'notebook_id' not in columns:
                # Add notebook_id column
                cursor.execute("""
                    ALTER TABLE conversations
                    ADD COLUMN notebook_id TEXT REFERENCES notebooks(id) ON DELETE CASCADE
                """)
                # Create index on notebook_id
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_notebook
                    ON conversations(notebook_id)
                """)

            # Migration: Add document_ids column if it doesn't exist
            if 'document_ids' not in columns:
                cursor.execute("""
                    ALTER TABLE conversations
                    ADD COLUMN document_ids TEXT
                """)

            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    is_error BOOLEAN DEFAULT 0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)

            # Create index on conversation_id and timestamp
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, timestamp ASC)
            """)

            # Create message_sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS message_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    document TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    relevance_score REAL NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                )
            """)

            # Create index on message_id
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_sources
                ON message_sources(message_id)
            """)

    def create_conversation(
        self,
        title: Optional[str] = None,
        notebook_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> str:
        """
        Create a new conversation.

        Args:
            title: Conversation title (defaults to "Nuova Conversazione")
            notebook_id: Optional notebook UUID to associate with
            document_ids: Optional list of document IDs for refinement mode

        Returns:
            conversation_id: UUID of created conversation
        """
        conversation_id = str(uuid.uuid4())
        now = datetime.now()

        # Serialize document_ids as JSON if provided
        doc_ids_json = json.dumps(document_ids) if document_ids else None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (id, title, created_at, updated_at, message_count, notebook_id, document_ids)
                VALUES (?, ?, ?, ?, 0, ?, ?)
            """, (conversation_id, title or "Nuova Conversazione", now, now, notebook_id, doc_ids_json))

        return conversation_id

    def list_conversations(
        self,
        notebook_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all conversations ordered by updated_at DESC.

        Args:
            notebook_id: Optional notebook filter

        Returns:
            List of conversation dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if notebook_id:
                cursor.execute("""
                    SELECT id, title, created_at, updated_at, message_count, notebook_id
                    FROM conversations
                    WHERE notebook_id = ?
                    ORDER BY updated_at DESC
                """, (notebook_id,))
            else:
                cursor.execute("""
                    SELECT id, title, created_at, updated_at, message_count, notebook_id
                    FROM conversations
                    ORDER BY updated_at DESC
                """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation metadata by ID.

        Args:
            conversation_id: Conversation UUID

        Returns:
            Conversation dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, created_at, updated_at, message_count, notebook_id, document_ids
                FROM conversations
                WHERE id = ?
            """, (conversation_id,))

            row = cursor.fetchone()
            if not row:
                return None

            result = dict(row)
            # Parse document_ids from JSON if present
            if result.get('document_ids'):
                result['document_ids'] = json.loads(result['document_ids'])
            return result

    def update_conversation_title(self, conversation_id: str, title: str):
        """
        Update conversation title.

        Args:
            conversation_id: Conversation UUID
            title: New title
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations
                SET title = ?
                WHERE id = ?
            """, (title, conversation_id))

    def delete_conversation(self, conversation_id: str):
        """
        Delete conversation and all its messages (CASCADE).

        Args:
            conversation_id: Conversation UUID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        is_error: bool = False,
        sources: Optional[List[Dict]] = None
    ) -> str:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation UUID
            role: 'user' or 'assistant'
            content: Message content
            is_error: Whether this is an error message
            sources: List of source dictionaries (for assistant messages)

        Returns:
            message_id: UUID of created message
        """
        message_id = str(uuid.uuid4())
        now = datetime.now()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Insert message
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, role, content, timestamp, is_error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (message_id, conversation_id, role, content, now, is_error))

            # Insert sources if provided
            if sources:
                for source in sources:
                    cursor.execute("""
                        INSERT INTO message_sources
                        (message_id, document, page, chunk_text, relevance_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        message_id,
                        source['document'],
                        source['page'],
                        source['chunk_text'],
                        source['relevance_score']
                    ))

            # Update conversation: increment message count and update timestamp
            cursor.execute("""
                UPDATE conversations
                SET message_count = message_count + 1,
                    updated_at = ?
                WHERE id = ?
            """, (now, conversation_id))

        return message_id

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation with sources.

        Args:
            conversation_id: Conversation UUID

        Returns:
            List of message dictionaries with sources
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get all messages
            cursor.execute("""
                SELECT id, conversation_id, role, content, timestamp, is_error
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """, (conversation_id,))

            messages = [dict(row) for row in cursor.fetchall()]

            # Get sources for each message
            for message in messages:
                cursor.execute("""
                    SELECT document, page, chunk_text, relevance_score
                    FROM message_sources
                    WHERE message_id = ?
                """, (message['id'],))

                message['sources'] = [dict(row) for row in cursor.fetchall()]

            return messages

    def get_conversation_with_messages(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation metadata with full message history.

        Args:
            conversation_id: Conversation UUID

        Returns:
            Dictionary with 'conversation' and 'messages' keys, or None if not found
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None

        messages = self.get_messages(conversation_id)

        # Return flat structure matching ConversationWithMessages model
        return {
            **conversation,
            "messages": messages
        }

    def list_conversations_by_notebook(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        List conversations for a specific notebook.

        Args:
            notebook_id: Notebook UUID

        Returns:
            List of conversation dictionaries
        """
        return self.list_conversations(notebook_id=notebook_id)


# Singleton instance
_conversation_db_instance = None


def get_conversation_db(db_path: str = "./conversations.db") -> ConversationDB:
    """
    Get singleton instance of ConversationDB.

    Args:
        db_path: Path to SQLite database file

    Returns:
        ConversationDB instance
    """
    global _conversation_db_instance
    if _conversation_db_instance is None:
        _conversation_db_instance = ConversationDB(db_path)
    return _conversation_db_instance
