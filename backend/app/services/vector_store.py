from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

settings = get_settings()


class VectorStore:
    """Vector store manager using ChromaDB."""

    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(
        self,
        document_id: str,
        document_name: str,
        chunks: List[Dict[str, Any]],
        notebook_id: Optional[str] = None
    ) -> int:
        """
        Add document chunks with embeddings to the vector store.

        Args:
            document_id: Unique document identifier
            document_name: Original document filename
            chunks: List of chunks with 'text', 'embedding', and 'metadata' fields
            notebook_id: Optional notebook UUID to associate document with

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        upload_timestamp = datetime.utcnow().isoformat()

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"{document_id}_{chunk['metadata']['chunk_index']}"
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])

            # Combine metadata with document info
            metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "page": chunk["metadata"]["page"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "upload_timestamp": upload_timestamp
            }
            # Store parent_chunk_index for parent-child retrieval
            if "parent_chunk_index" in chunk["metadata"]:
                metadata["parent_chunk_index"] = chunk["metadata"]["parent_chunk_index"]
            # Only add notebook_id if it's not None (ChromaDB doesn't accept None values)
            if notebook_id is not None:
                metadata["notebook_id"] = notebook_id
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        document_ids: Optional[List[str]] = None,
        notebook_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for similar chunks in the vector store.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            document_ids: Optional list of document IDs to filter by
            notebook_id: Optional notebook ID to filter by

        Returns:
            Dictionary with ids, documents, metadatas, and distances
        """
        # Build where clause for filtering
        where = None

        if notebook_id and document_ids:
            # Filter by both notebook and specific documents
            where = {
                "$and": [
                    {"notebook_id": notebook_id},
                    {"document_id": {"$in": document_ids}}
                ]
            }
        elif notebook_id:
            # Filter only by notebook
            where = {"notebook_id": notebook_id}
        elif document_ids:
            # Filter only by documents (existing logic)
            if len(document_ids) == 1:
                where = {"document_id": document_ids[0]}
            else:
                where = {"document_id": {"$in": document_ids}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        return results

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks of a document from the vector store.

        Args:
            document_id: Document identifier

        Returns:
            Number of chunks deleted
        """
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )

        if results and results["ids"]:
            chunk_ids = results["ids"]
            self.collection.delete(ids=chunk_ids)
            return len(chunk_ids)

        return 0

    def list_documents(self, notebook_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all documents in the vector store.

        Args:
            notebook_id: Optional notebook ID to filter by

        Returns:
            List of document information dictionaries
        """
        # Get items from collection with optional filter
        if notebook_id:
            all_items = self.collection.get(where={"notebook_id": notebook_id})
        else:
            all_items = self.collection.get()

        if not all_items or not all_items["metadatas"]:
            return []

        # Group by document_id and aggregate info
        documents_map = {}

        for metadata in all_items["metadatas"]:
            doc_id = metadata.get("document_id")
            if doc_id not in documents_map:
                documents_map[doc_id] = {
                    "document_id": doc_id,
                    "document_name": metadata.get("document_name"),
                    "upload_timestamp": metadata.get("upload_timestamp"),
                    "chunk_count": 1,
                    "max_page": metadata.get("page", 1),
                    "notebook_id": metadata.get("notebook_id")  # NUOVO
                }
            else:
                documents_map[doc_id]["chunk_count"] += 1
                documents_map[doc_id]["max_page"] = max(
                    documents_map[doc_id]["max_page"],
                    metadata.get("page", 1)
                )

        # Convert to list and add page_count
        documents = []
        for doc_info in documents_map.values():
            doc_info["page_count"] = doc_info.pop("max_page")
            documents.append(doc_info)

        # Sort by upload timestamp (most recent first)
        documents.sort(key=lambda x: x.get("upload_timestamp", ""), reverse=True)

        return documents

    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the vector store.

        Args:
            document_id: Document identifier

        Returns:
            True if document exists, False otherwise
        """
        results = self.collection.get(
            where={"document_id": document_id},
            limit=1
        )

        return bool(results and results["ids"])

    def update_document_notebook(
        self,
        document_id: str,
        notebook_id: Optional[str]
    ) -> int:
        """
        Update the notebook_id for all chunks of a document.

        Args:
            document_id: Document identifier
            notebook_id: New notebook UUID (None to unassign)

        Returns:
            Number of chunks updated
        """
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )

        if not results or not results["ids"]:
            return 0

        chunk_ids = results["ids"]
        metadatas = results["metadatas"]

        # Update notebook_id in metadata
        updated_metadatas = []
        for metadata in metadatas:
            # Only include notebook_id if not None (ChromaDB doesn't accept None values)
            if notebook_id is not None:
                metadata["notebook_id"] = notebook_id
            elif "notebook_id" in metadata:
                # Remove notebook_id if setting to None
                del metadata["notebook_id"]
            updated_metadatas.append(metadata)

        # Update metadata in-place (preserves existing embeddings)
        self.collection.update(
            ids=chunk_ids,
            metadatas=updated_metadatas
        )

        return len(chunk_ids)

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }

    def get_document_chunks(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get all chunks for specified documents.

        Args:
            document_ids: List of document IDs to retrieve chunks for

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not document_ids:
            return []

        # Build where clause
        if len(document_ids) == 1:
            where = {"document_id": document_ids[0]}
        else:
            where = {"document_id": {"$in": document_ids}}

        results = self.collection.get(where=where)

        if not results or not results["ids"]:
            return []

        chunks = []
        for i, doc_id in enumerate(results["ids"]):
            chunks.append({
                "id": doc_id,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })

        # Sort by document_name, then page, then chunk_index
        chunks.sort(key=lambda x: (
            x["metadata"].get("document_name", ""),
            x["metadata"].get("page", 0),
            x["metadata"].get("chunk_index", 0)
        ))

        return chunks


# Singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create vector store singleton instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
