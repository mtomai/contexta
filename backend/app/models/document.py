from pydantic import BaseModel
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    document_name: str
    page_count: int
    chunk_count: int
    upload_timestamp: datetime
    notebook_id: str | None = None


class DocumentInfo(BaseModel):
    """Model for document information."""
    document_id: str
    document_name: str
    upload_timestamp: datetime
    page_count: int
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: list[DocumentInfo]
    total: int
