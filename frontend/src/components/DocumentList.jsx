import React, { useState, useEffect } from 'react';
import { FileText, Trash2, Calendar, FileCheck, CheckSquare, Square } from 'lucide-react';
import { getDocuments, deleteDocument } from '../services/api';

const DocumentList = ({ refreshTrigger, notebookId, selectedDocuments = [], onSelectionChange }) => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  useEffect(() => {
    fetchDocuments();
  }, [refreshTrigger, notebookId]);

  // Clean up selected documents when documents change
  useEffect(() => {
    if (onSelectionChange && documents.length > 0) {
      const validIds = documents.map(d => d.document_id);
      const cleanedSelection = selectedDocuments.filter(id => validIds.includes(id));
      if (cleanedSelection.length !== selectedDocuments.length) {
        onSelectionChange(cleanedSelection);
      }
    }
  }, [documents]);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      console.debug('[DocumentList] Fetching documents for notebook:', notebookId);
      const data = await getDocuments(notebookId);
      console.debug('[DocumentList] Received', data?.documents?.length ?? 0, 'documents');
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('[DocumentList] Error fetching documents:', err);
      setError('Error loading documents');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (documentId, documentName) => {
    if (!window.confirm(`Are you sure you want to delete "${documentName}"?`)) {
      return;
    }

    try {
      setDeletingId(documentId);
      await deleteDocument(documentId);
      setDocuments(documents.filter(doc => doc.document_id !== documentId));
      // Remove from selection if selected
      if (onSelectionChange && selectedDocuments.includes(documentId)) {
        onSelectionChange(selectedDocuments.filter(id => id !== documentId));
      }
    } catch (err) {
      alert('Error deleting document');
      console.error(err);
    } finally {
      setDeletingId(null);
    }
  };

  const handleToggleSelect = (documentId) => {
    if (!onSelectionChange) return;

    if (selectedDocuments.includes(documentId)) {
      onSelectionChange(selectedDocuments.filter(id => id !== documentId));
    } else {
      onSelectionChange([...selectedDocuments, documentId]);
    }
  };

  const handleSelectAll = () => {
    if (!onSelectionChange) return;

    if (selectedDocuments.length === documents.length) {
      onSelectionChange([]);
    } else {
      onSelectionChange(documents.map(d => d.document_id));
    }
  };

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const isSelected = (documentId) => selectedDocuments.includes(documentId);

  if (loading) {
    return (
      <div style={styles.container}>
        <h2 style={styles.title}>Your Documents</h2>
        <p style={styles.loadingText}>Loading...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.container}>
        <h2 style={styles.title}>Your Documents</h2>
        <p style={styles.errorText}>{error}</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <h2 style={styles.title}>Your Documents</h2>
        {documents.length > 0 && onSelectionChange && (
          <button onClick={handleSelectAll} style={styles.selectAllButton}>
            {selectedDocuments.length === documents.length ? (
              <CheckSquare size={16} />
            ) : (
              <Square size={16} />
            )}
            <span>{selectedDocuments.length === documents.length ? 'Deselect' : 'Select all'}</span>
          </button>
        )}
      </div>

      {selectedDocuments.length > 0 && (
        <div style={styles.selectionInfo}>
          {selectedDocuments.length} document{selectedDocuments.length !== 1 ? 's' : ''} selected
        </div>
      )}

      {documents.length === 0 ? (
        <p style={styles.emptyText}>No documents uploaded yet.</p>
      ) : (
        <div style={styles.documentList}>
          {documents.map((doc) => (
            <div
              key={doc.document_id}
              style={{
                ...styles.documentCard,
                ...(isSelected(doc.document_id) ? styles.documentCardSelected : {})
              }}
              onClick={() => handleToggleSelect(doc.document_id)}
            >
              {onSelectionChange && (
                <div style={styles.checkboxContainer}>
                  {isSelected(doc.document_id) ? (
                    <CheckSquare size={20} color="#2563eb" />
                  ) : (
                    <Square size={20} color="#9ca3af" />
                  )}
                </div>
              )}

              <div style={styles.documentIcon}>
                <FileText size={32} color="#4a90e2" />
              </div>

              <div style={styles.documentInfo}>
                <h3 style={styles.documentName}>{doc.document_name}</h3>

                <div style={styles.documentMeta}>
                  <div style={styles.metaItem}>
                    <Calendar size={14} />
                    <span>{formatDate(doc.upload_timestamp)}</span>
                  </div>
                  <div style={styles.metaItem}>
                    <FileCheck size={14} />
                    <span>{doc.page_count} pages</span>
                  </div>
                  <div style={styles.metaItem}>
                    <FileText size={14} />
                    <span>{doc.chunk_count} chunks</span>
                  </div>
                </div>
              </div>

              <button
                style={{
                  ...styles.deleteButton,
                  ...(deletingId === doc.document_id ? styles.deleteButtonDisabled : {})
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(doc.document_id, doc.document_name);
                }}
                disabled={deletingId === doc.document_id}
              >
                <Trash2 size={18} />
              </button>
            </div>
          ))}
        </div>
      )}

      {documents.length > 0 && (
        <div style={styles.summary}>
          Total: {documents.length} document{documents.length !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    padding: '20px',
    backgroundColor: '#fff',
    borderRadius: '8px',
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    minHeight: 0,
    overflow: 'hidden',
  },
  headerRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
    flexShrink: 0,
  },
  title: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#333',
    margin: 0,
  },
  selectAllButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '6px 10px',
    border: '1px solid #e0e0e0',
    borderRadius: '6px',
    backgroundColor: 'white',
    cursor: 'pointer',
    fontSize: '12px',
    color: '#6b7280',
    transition: 'all 0.2s',
  },
  selectionInfo: {
    padding: '8px 12px',
    backgroundColor: '#eff6ff',
    borderRadius: '6px',
    fontSize: '13px',
    color: '#2563eb',
    marginBottom: '12px',
    textAlign: 'center',
    flexShrink: 0,
  },
  loadingText: {
    color: '#666',
    textAlign: 'center',
    padding: '20px',
  },
  errorText: {
    color: '#f44336',
    textAlign: 'center',
    padding: '20px',
  },
  emptyText: {
    color: '#999',
    textAlign: 'center',
    padding: '40px 20px',
    fontStyle: 'italic',
  },
  documentList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    flex: 1,
    overflowY: 'auto',
    minHeight: 0,
    paddingRight: '4px',
  },
  documentCard: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '12px',
    border: '1px solid #e0e0e0',
    borderRadius: '6px',
    backgroundColor: '#fafafa',
    transition: 'all 0.2s',
    cursor: 'pointer',
    flexShrink: 0,
  },
  documentCardSelected: {
    borderColor: '#2563eb',
    backgroundColor: '#eff6ff',
  },
  checkboxContainer: {
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  documentIcon: {
    flexShrink: 0,
  },
  documentInfo: {
    flex: 1,
    minWidth: 0,
  },
  documentName: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#333',
    marginBottom: '6px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  documentMeta: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  metaItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '12px',
    color: '#666',
  },
  deleteButton: {
    flexShrink: 0,
    padding: '8px',
    border: 'none',
    backgroundColor: 'transparent',
    color: '#f44336',
    cursor: 'pointer',
    borderRadius: '4px',
    transition: 'background-color 0.2s',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  deleteButtonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  summary: {
    marginTop: '16px',
    paddingTop: '12px',
    borderTop: '1px solid #e0e0e0',
    fontSize: '14px',
    color: '#666',
    textAlign: 'center',
    flexShrink: 0,
  },
};

export default DocumentList;
