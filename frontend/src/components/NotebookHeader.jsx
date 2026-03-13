import { useState, useEffect } from 'react';
import { ArrowLeft, BookOpen, Edit2 } from 'lucide-react';

function NotebookHeader({ notebookId, onBack }) {
  const [notebook, setNotebook] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDescription, setNewDescription] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchNotebook();
  }, [notebookId]);

  const fetchNotebook = async () => {
    try {
      setLoading(true);
      const api = await import('../services/api.js');
      const data = await api.getNotebook(notebookId);
      setNotebook(data);
      setNewName(data.name);
      setNewDescription(data.description || '');
    } catch (err) {
      console.error('Error fetching notebook:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRename = async () => {
    if (!newName.trim()) {
      alert('Enter a name for the notebook');
      return;
    }

    try {
      setSaving(true);
      const api = await import('../services/api.js');
      await api.updateNotebook(notebookId, newName, newDescription || null);
      setShowRenameModal(false);
      fetchNotebook();
    } catch (err) {
      console.error('Error updating notebook:', err);
      alert('Error updating notebook');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div style={styles.header}>
        <div style={styles.loading}>Loading...</div>
      </div>
    );
  }

  return (
    <>
      <div style={styles.header}>
        <button style={styles.backButton} onClick={onBack}>
          <ArrowLeft size={20} />
          Notebooks
        </button>

        <div style={styles.titleContainer}>
          <BookOpen size={20} style={styles.icon} />
          <h2 style={styles.title}>{notebook?.name}</h2>
          <button
            style={styles.editButton}
            onClick={() => setShowRenameModal(true)}
            title="Rename notebook"
          >
            <Edit2 size={16} />
          </button>
        </div>
      </div>

      {/* Rename Modal */}
      {showRenameModal && (
        <div style={styles.modalOverlay} onClick={() => setShowRenameModal(false)}>
          <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
            <h2 style={styles.modalTitle}>Edit Notebook</h2>

            <div style={styles.inputGroup}>
              <label style={styles.label}>Name *</label>
              <input
                type="text"
                style={styles.input}
                placeholder="Notebook name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                autoFocus
              />
            </div>

            <div style={styles.inputGroup}>
              <label style={styles.label}>Description</label>
              <textarea
                style={{...styles.input, ...styles.textarea}}
                placeholder="Notebook description..."
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                rows={3}
              />
            </div>

            <div style={styles.modalActions}>
              <button
                style={styles.cancelButton}
                onClick={() => setShowRenameModal(false)}
                disabled={saving}
              >
                Cancel
              </button>
              <button
                style={styles.confirmButton}
                onClick={handleRename}
                disabled={saving || !newName.trim()}
              >
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

const styles = {
  header: {
    backgroundColor: '#f8f9fa',
    borderBottom: '1px solid #e0e0e0',
    padding: '16px 24px',
  },
  loading: {
    color: '#666',
    fontSize: '14px',
  },
  backButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 16px',
    backgroundColor: 'transparent',
    border: '1px solid #ddd',
    borderRadius: '6px',
    fontSize: '14px',
    color: '#666',
    cursor: 'pointer',
    marginBottom: '12px',
    transition: 'all 0.2s',
  },
  titleContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  icon: {
    color: '#4a90e2',
  },
  title: {
    fontSize: '20px',
    fontWeight: '600',
    color: '#1a1a1a',
    margin: 0,
    flex: 1,
  },
  editButton: {
    background: 'none',
    border: 'none',
    color: '#999',
    cursor: 'pointer',
    padding: '8px',
    borderRadius: '6px',
    transition: 'all 0.2s',
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

export default NotebookHeader;
