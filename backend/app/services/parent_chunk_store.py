"""
Parent Chunk Store Module

Persistent SQLite storage for parent chunks in the Parent-Child Retrieval strategy.
Parent chunks are large text blocks (~1500 tokens) that provide broad context for the LLM,
while their corresponding child chunks (~300 tokens) are stored in ChromaDB for precise search.
"""

import sqlite3
import threading
from typing import List, Dict, Any, Optional

from app.config import get_settings

settings = get_settings()


class ParentChunkStore:
    """Thread-safe SQLite store for parent chunks."""

    def __init__(self, db_path: str = None):
        """
        Initialize the parent chunk store.

        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = "./parent_chunks.db"

        self._db_path = db_path
        self._lock = threading.Lock()
        self._create_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def _create_table(self):
        """Create the parent_chunks table if it doesn't exist."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    parent_index INTEGER NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    notebook_id TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_chunks_document
                ON parent_chunks(document_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_chunks_lookup
                ON parent_chunks(document_id, parent_index)
            """)
            conn.commit()
        finally:
            conn.close()

    def add_parent_chunks(
        self,
        document_id: str,
        document_name: str,
        parent_chunks: List[Dict[str, Any]],
        notebook_id: Optional[str] = None
    ) -> int:
        """
        Store parent chunks for a document.

        Args:
            document_id: Document UUID
            document_name: Original filename
            parent_chunks: List of parent chunk dicts with 'text' and 'metadata'
            notebook_id: Optional notebook UUID

        Returns:
            Number of parent chunks stored
        """
        if not parent_chunks:
            return 0

        with self._lock:
            conn = self._get_connection()
            try:
                for chunk in parent_chunks:
                    metadata = chunk.get("metadata", {})
                    parent_index = metadata.get("parent_chunk_index", 0)
                    page = metadata.get("page", 1)
                    chunk_id = f"{document_id}_parent_{parent_index}"

                    conn.execute(
                        """INSERT OR REPLACE INTO parent_chunks
                           (id, document_id, document_name, parent_index, page, text, notebook_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (chunk_id, document_id, document_name, parent_index, page,
                         chunk["text"], notebook_id)
                    )
                conn.commit()
                return len(parent_chunks)
            finally:
                conn.close()

    def get_parent_chunk(
        self,
        document_id: str,
        parent_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single parent chunk by document ID and parent index.

        Args:
            document_id: Document UUID
            parent_index: Parent chunk index

        Returns:
            Parent chunk dict or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT id, text, page, document_name, notebook_id
                   FROM parent_chunks
                   WHERE document_id = ? AND parent_index = ?""",
                (document_id, parent_index)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "text": row[1],
                    "page": row[2],
                    "document_name": row[3],
                    "notebook_id": row[4]
                }
            return None
        finally:
            conn.close()

    def get_parent_chunks_batch(
        self,
        lookups: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve multiple parent chunks efficiently.

        Args:
            lookups: List of dicts with 'document_id' and 'parent_index' keys

        Returns:
            Dict mapping "document_id:parent_index" to parent chunk data
        """
        if not lookups:
            return {}

        results = {}
        conn = self._get_connection()
        try:
            for lookup in lookups:
                doc_id = lookup["document_id"]
                parent_idx = lookup["parent_index"]
                key = f"{doc_id}:{parent_idx}"

                cursor = conn.execute(
                    """SELECT id, text, page, document_name
                       FROM parent_chunks
                       WHERE document_id = ? AND parent_index = ?""",
                    (doc_id, parent_idx)
                )
                row = cursor.fetchone()
                if row:
                    results[key] = {
                        "id": row[0],
                        "text": row[1],
                        "page": row[2],
                        "document_name": row[3]
                    }
            return results
        finally:
            conn.close()

    def delete_document(self, document_id: str) -> int:
        """
        Delete all parent chunks for a document.

        Args:
            document_id: Document UUID

        Returns:
            Number of parent chunks deleted
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM parent_chunks WHERE document_id = ?",
                    (document_id,)
                )
                count = cursor.fetchone()[0]

                conn.execute(
                    "DELETE FROM parent_chunks WHERE document_id = ?",
                    (document_id,)
                )
                conn.commit()
                return count
            finally:
                conn.close()

    def update_notebook(self, document_id: str, notebook_id: Optional[str]) -> int:
        """
        Update notebook_id for all parent chunks of a document.

        Args:
            document_id: Document UUID
            notebook_id: New notebook UUID (None to unassign)

        Returns:
            Number of parent chunks updated
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "UPDATE parent_chunks SET notebook_id = ? WHERE document_id = ?",
                    (notebook_id, document_id)
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    def get_document_parent_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all parent chunks for a document, sorted by parent_index.

        Args:
            document_id: Document UUID

        Returns:
            List of parent chunk dictionaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT id, text, page, document_name, parent_index, notebook_id
                   FROM parent_chunks
                   WHERE document_id = ?
                   ORDER BY parent_index""",
                (document_id,)
            )
            return [
                {
                    "id": row[0],
                    "text": row[1],
                    "page": row[2],
                    "document_name": row[3],
                    "parent_index": row[4],
                    "notebook_id": row[5]
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_documents_parent_chunks_batch(
        self,
        document_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all parent chunks for multiple documents in a single query.

        Uses an SQL ``IN`` clause to avoid the N+1 query problem when
        fetching chunks for many documents at once.

        Args:
            document_ids: List of document UUIDs to fetch.

        Returns:
            List of parent chunk dicts, ordered by document_id and parent_index.
        """
        if not document_ids:
            return []

        conn = self._get_connection()
        try:
            placeholders = ",".join("?" for _ in document_ids)
            cursor = conn.execute(
                f"""SELECT id, document_id, text, page, document_name,
                           parent_index, notebook_id
                    FROM parent_chunks
                    WHERE document_id IN ({placeholders})
                    ORDER BY document_id, parent_index""",
                document_ids,
            )
            return [
                {
                    "id": row[0],
                    "document_id": row[1],
                    "text": row[2],
                    "page": row[3],
                    "document_name": row[4],
                    "parent_index": row[5],
                    "notebook_id": row[6],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()


# Singleton instance
_parent_chunk_store: Optional[ParentChunkStore] = None


def get_parent_chunk_store() -> ParentChunkStore:
    """Get or create parent chunk store singleton instance."""
    global _parent_chunk_store
    if _parent_chunk_store is None:
        _parent_chunk_store = ParentChunkStore()
    return _parent_chunk_store
