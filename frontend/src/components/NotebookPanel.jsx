import { useState, useEffect } from 'react';
import { Plus, BookOpen, Trash2, Calendar, FileText, MessageSquare } from 'lucide-react';

function NotebookPanel({ selectedNotebookId, onSelectNotebook, refreshTrigger }) {
  const [notebooks, setNotebooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newNotebookName, setNewNotebookName] = useState('');
  const [newNotebookDescription, setNewNotebookDescription] = useState('');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    fetchNotebooks();
  }, [refreshTrigger]);

  const fetchNotebooks = async () => {
    try {
      setLoading(true);
      setError(null);

      // Importa dinamicamente per evitare circular dependencies
      const api = await import('../services/api.js');
      const data = await api.getNotebooks();
      setNotebooks(data || []);
    } catch (err) {
      console.error('Error fetching notebooks:', err);
      setError('Error loading notebooks');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateNotebook = async () => {
    if (!newNotebookName.trim()) {
      alert('Enter a name for the notebook');
      return;
    }

    try {
      setCreating(true);
      const api = await import('../services/api.js');
      await api.createNotebook(newNotebookName, newNotebookDescription || null);

      setNewNotebookName('');
      setNewNotebookDescription('');
      setShowCreateModal(false);
      fetchNotebooks();
    } catch (err) {
      console.error('Error creating notebook:', err);
      alert('Error creating notebook');
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteNotebook = async (notebookId, notebookName) => {
    if (!confirm(`Delete notebook "${notebookName}"? Conversations will be deleted but documents will remain available.`)) {
      return;
    }

    try {
      const api = await import('../services/api.js');
      await api.deleteNotebook(notebookId);
      fetchNotebooks();
    } catch (err) {
      console.error('Error deleting notebook:', err);
      alert('Error deleting notebook');
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

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Your Notebooks</h1>
        <button
          style={styles.createButton}
          onClick={() => setShowCreateModal(true)}
        >
          <Plus size={20} />
          New Notebook
        </button>
      </div>

      {error && (
        <div style={styles.error}>{error}</div>
      )}

      {loading ? (
        <div style={styles.loading}>Loading notebooks...</div>
      ) : notebooks.length === 0 ? (
        <div style={styles.empty}>
          <BookOpen size={64} style={styles.emptyIcon} />
          <h2>No notebooks found</h2>
          <p>Create your first notebook to get started</p>
          <button
            style={styles.createButtonLarge}
            onClick={() => setShowCreateModal(true)}
          >
            <Plus size={20} />
            Create Notebook
          </button>
        </div>
      ) : (
        <div style={styles.grid}>
          {notebooks.map((notebook) => (
            <div
              key={notebook.id}
              style={styles.card}
              onClick={() => onSelectNotebook(notebook.id)}
            >
              <div style={styles.cardHeader}>
                <BookOpen size={24} style={styles.cardIcon} />
                <button
                  style={styles.deleteButton}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteNotebook(notebook.id, notebook.name);
                  }}
                >
                  <Trash2 size={16} />
                </button>
              </div>

              <h3 style={styles.cardTitle}>{notebook.name}</h3>

              {notebook.description && (
                <p style={styles.cardDescription}>{notebook.description}</p>
              )}

              <div style={styles.cardStats}>
                <div style={styles.stat}>
                  <FileText size={14} />
                  <span>{notebook.document_count || 0} documents</span>
                </div>
                <div style={styles.stat}>
                  <MessageSquare size={14} />
                  <span>{notebook.conversation_count || 0} chats</span>
                </div>
              </div>

              <div style={styles.cardFooter}>
                <Calendar size={12} />
                <span style={styles.date}>{formatDate(notebook.updated_at)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div style={styles.modalOverlay} onClick={() => setShowCreateModal(false)}>
          <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
            <h2 style={styles.modalTitle}>Create New Notebook</h2>

            <div style={styles.inputGroup}>
              <label style={styles.label}>Name *</label>
              <input
                type="text"
                style={styles.input}
                placeholder="Ex: Work Notes"
                value={newNotebookName}
                onChange={(e) => setNewNotebookName(e.target.value)}
                autoFocus
              />
            </div>

            <div style={styles.inputGroup}>
              <label style={styles.label}>Description (optional)</label>
              <textarea
                style={{...styles.input, ...styles.textarea}}
                placeholder="Describe the content of this notebook..."
                value={newNotebookDescription}
                onChange={(e) => setNewNotebookDescription(e.target.value)}
                rows={3}
              />
            </div>

            <div style={styles.modalActions}>
              <button
                style={styles.cancelButton}
                onClick={() => setShowCreateModal(false)}
                disabled={creating}
              >
                Cancel
              </button>
              <button
                style={styles.confirmButton}
                onClick={handleCreateNotebook}
                disabled={creating || !newNotebookName.trim()}
              >
                {creating ? 'Creating...' : 'Create Notebook'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '40px 20px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '32px',
  },
  title: {
    fontSize: '32px',
    fontWeight: '600',
    color: '#1a1a1a',
    margin: 0,
  },
  createButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '12px 24px',
    backgroundColor: '#4a90e2',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  loading: {
    textAlign: 'center',
    padding: '60px 20px',
    color: '#666',
    fontSize: '18px',
  },
  error: {
    padding: '16px',
    backgroundColor: '#fee',
    color: '#c00',
    borderRadius: '8px',
    marginBottom: '20px',
  },
  empty: {
    textAlign: 'center',
    padding: '80px 20px',
  },
  emptyIcon: {
    color: '#ddd',
    marginBottom: '20px',
  },
  createButtonLarge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    padding: '16px 32px',
    backgroundColor: '#4a90e2',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '18px',
    cursor: 'pointer',
    marginTop: '20px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    gap: '24px',
  },
  card: {
    backgroundColor: 'white',
    border: '1px solid #e0e0e0',
    borderRadius: '12px',
    padding: '24px',
    cursor: 'pointer',
    transition: 'all 0.2s',
    '&:hover': {
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
      transform: 'translateY(-2px)',
    },
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'start',
    marginBottom: '16px',
  },
  cardIcon: {
    color: '#4a90e2',
  },
  deleteButton: {
    background: 'none',
    border: 'none',
    color: '#999',
    cursor: 'pointer',
    padding: '4px',
    transition: 'color 0.2s',
  },
  cardTitle: {
    fontSize: '20px',
    fontWeight: '600',
    color: '#1a1a1a',
    marginBottom: '8px',
  },
  cardDescription: {
    fontSize: '14px',
    color: '#666',
    marginBottom: '16px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    display: '-webkit-box',
    WebkitLineClamp: 2,
    WebkitBoxOrient: 'vertical',
  },
  cardStats: {
    display: 'flex',
    gap: '16px',
    marginBottom: '16px',
    paddingTop: '16px',
    borderTop: '1px solid #f0f0f0',
  },
  stat: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '14px',
    color: '#666',
  },
  cardFooter: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '12px',
    color: '#999',
  },
  date: {
    fontSize: '12px',
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  modal: {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '32px',
    width: '90%',
    maxWidth: '500px',
  },
  modalTitle: {
    fontSize: '24px',
    fontWeight: '600',
    marginBottom: '24px',
    color: '#1a1a1a',
  },
  inputGroup: {
    marginBottom: '20px',
  },
  label: {
    display: 'block',
    fontSize: '14px',
    fontWeight: '500',
    marginBottom: '8px',
    color: '#333',
  },
  input: {
    width: '100%',
    padding: '12px',
    fontSize: '16px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    boxSizing: 'border-box',
  },
  textarea: {
    resize: 'vertical',
    fontFamily: 'inherit',
  },
  modalActions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '12px',
    marginTop: '24px',
  },
  cancelButton: {
    padding: '12px 24px',
    backgroundColor: '#f5f5f5',
    color: '#333',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    cursor: 'pointer',
  },
  confirmButton: {
    padding: '12px 24px',
    backgroundColor: '#4a90e2',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    cursor: 'pointer',
  },
};

export default NotebookPanel;
