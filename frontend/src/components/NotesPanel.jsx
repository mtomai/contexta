import { useState, useEffect } from 'react';
import { Pin, Trash2, FileText, Loader, Eye, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { getNotes, deleteNote } from '../services/api';

function NotesPanel({ notebookId, refreshTrigger }) {
  const [notes, setNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewingNote, setViewingNote] = useState(null);

  useEffect(() => {
    if (notebookId) {
      fetchNotes();
    }
  }, [notebookId, refreshTrigger]);

  const fetchNotes = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getNotes(notebookId);
      setNotes(data || []);
    } catch (err) {
      console.error('Error fetching notes:', err);
      setError('Error loading notes');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (noteId) => {
    if (!confirm('Delete this note?')) return;
    try {
      await deleteNote(noteId);
      setNotes(prev => prev.filter(n => n.id !== noteId));
    } catch (err) {
      console.error('Error deleting note:', err);
      alert('Error deleting note');
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US');
  };

  const truncate = (text, maxLen = 200) => {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen).trimEnd() + '…';
  };

  if (loading) {
    return (
      <div style={styles.centered}>
        <Loader size={24} style={{ animation: 'spin 1s linear infinite' }} />
        <span style={{ marginLeft: 8, color: '#666' }}>Loading notes...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.centered}>
        <p style={{ color: '#c62828' }}>{error}</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <Pin size={18} />
        <span style={styles.title}>Saved Notes</span>
        <span style={styles.badge}>{notes.length}</span>
      </div>

      {notes.length === 0 ? (
        <div style={styles.empty}>
          <FileText size={40} color="#ccc" />
          <p style={styles.emptyText}>No notes saved</p>
          <p style={styles.emptySubtext}>
            Click "Save note" on an AI response to save it here
          </p>
        </div>
      ) : (
        <div style={styles.list}>
          {notes.map((note) => (
            <div key={note.id} style={styles.card}>
              <p style={styles.cardContent}>{truncate(note.content)}</p>
              <div style={styles.cardFooter}>
                <span style={styles.cardDate}>{formatDate(note.created_at)}</span>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button
                    onClick={() => setViewingNote(note)}
                    style={styles.viewBtn}
                    title="View note"
                  >
                    <Eye size={14} />
                  </button>
                  <button
                    onClick={() => handleDelete(note.id)}
                    style={styles.deleteBtn}
                    title="Delete note"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal per visualizzare la nota in markdown */}
      {viewingNote && (
        <div style={styles.modalOverlay} onClick={() => setViewingNote(null)}>
          <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <span style={styles.modalTitle}>Note</span>
              <button
                onClick={() => setViewingNote(null)}
                style={styles.modalCloseBtn}
                title="Close"
              >
                <X size={18} />
              </button>
            </div>
            <div style={styles.modalBody}>
              <ReactMarkdown>{viewingNote.content}</ReactMarkdown>
            </div>
            <div style={styles.modalFooter}>
              <span style={styles.cardDate}>{formatDate(viewingNote.created_at)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  },
  centered: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px 20px',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '16px',
    borderBottom: '1px solid #e0e0e0',
    fontWeight: '600',
    fontSize: '15px',
    color: '#333',
    flexShrink: 0,
  },
  title: {
    flex: 1,
  },
  badge: {
    backgroundColor: '#e3f2fd',
    color: '#1976d2',
    fontSize: '12px',
    fontWeight: '600',
    padding: '2px 8px',
    borderRadius: '12px',
  },
  empty: {
    textAlign: 'center',
    padding: '40px 20px',
    color: '#666',
  },
  emptyText: {
    fontSize: '15px',
    marginTop: '12px',
    marginBottom: '4px',
    fontWeight: '500',
  },
  emptySubtext: {
    fontSize: '13px',
    color: '#999',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    padding: '8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  card: {
    backgroundColor: '#fffde7',
    border: '1px solid #fff9c4',
    borderRadius: '8px',
    padding: '12px',
  },
  cardContent: {
    fontSize: '13px',
    lineHeight: '1.5',
    color: '#333',
    margin: 0,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  cardFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: '8px',
  },
  cardDate: {
    fontSize: '11px',
    color: '#999',
  },
  deleteBtn: {
    background: 'none',
    border: 'none',
    color: '#999',
    cursor: 'pointer',
    padding: '4px',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    transition: 'color 0.2s',
  },
  viewBtn: {
    background: 'none',
    border: 'none',
    color: '#999',
    cursor: 'pointer',
    padding: '4px',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    transition: 'color 0.2s',
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: '12px',
    width: '90%',
    maxWidth: '700px',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
  },
  modalHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 20px',
    borderBottom: '1px solid #e0e0e0',
    flexShrink: 0,
  },
  modalTitle: {
    fontWeight: '600',
    fontSize: '16px',
    color: '#333',
  },
  modalCloseBtn: {
    background: 'none',
    border: 'none',
    color: '#666',
    cursor: 'pointer',
    padding: '4px',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
  },
  modalBody: {
    padding: '20px',
    overflowY: 'auto',
    flex: 1,
    fontSize: '14px',
    lineHeight: '1.7',
    color: '#333',
  },
  modalFooter: {
    padding: '12px 20px',
    borderTop: '1px solid #e0e0e0',
    flexShrink: 0,
  },
};

export default NotesPanel;
