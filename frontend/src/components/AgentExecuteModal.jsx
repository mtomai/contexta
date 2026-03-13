import React, { useState, useEffect } from 'react';
import { X, Play, FileText } from 'lucide-react';

const AgentExecuteModal = ({
  isOpen,
  onClose,
  agent,
  selectedDocuments,
  notebookId,
  onExecuteStream
}) => {
  const [variableValues, setVariableValues] = useState({});
  const [error, setError] = useState(null);

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen && agent) {
      // Initialize with defaults
      const defaults = {};
      (agent.variables || []).forEach(v => {
        defaults[v.key] = v.default || '';
      });
      setVariableValues(defaults);
      setError(null);
    }
  }, [isOpen, agent]);

  const handleVariableChange = (key, value) => {
    setVariableValues(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleExecute = () => {
    // Validate required variables
    for (const v of (agent?.variables || [])) {
      if (v.required && !variableValues[v.key]) {
        setError(`The variable "${v.label}" is required`);
        return;
      }
    }

    // Delegate streaming execution to parent and close immediately
    if (onExecuteStream) {
      onExecuteStream(agent, selectedDocuments, variableValues);
    }
    onClose();
  };

  const renderVariableInput = (variable) => {
    const value = variableValues[variable.key] || '';

    return (
      <input
        type="text"
        value={value}
        onChange={e => handleVariableChange(variable.key, e.target.value)}
        style={styles.input}
        placeholder={variable.placeholder || ''}
      />
    );
  };

  if (!isOpen || !agent) return null;

  const hasVariables = agent.variables && agent.variables.length > 0;

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>Execute: {agent.name}</h2>
          <button onClick={onClose} style={styles.closeButton}>
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div style={styles.content}>
          {error && (
            <div style={styles.error}>{error}</div>
          )}

          {/* Document Info */}
          <div style={styles.infoBox}>
            <FileText size={16} />
            <span>{selectedDocuments.length} document(s) selected</span>
          </div>

          {/* Agent Description */}
          {agent.description && (
            <p style={styles.description}>{agent.description}</p>
          )}

          {/* Variables Section */}
          {hasVariables && (
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>Variables</h3>
              <div style={styles.variablesList}>
                {agent.variables.map(variable => (
                  <div key={variable.key} style={styles.variableField}>
                    <label style={styles.label}>
                      {variable.label}
                      {variable.required && <span style={styles.required}>*</span>}
                    </label>
                    {renderVariableInput(variable)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={styles.footer}>
          <button
            onClick={onClose}
            style={styles.cancelButton}
          >
            Cancel
          </button>
          <button
            onClick={handleExecute}
            style={styles.executeButton}
          >
            <Play size={16} />
            Execute
          </button>
        </div>
      </div>
    </div>
  );
};

const styles = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modal: {
    backgroundColor: 'white',
    borderRadius: '12px',
    width: '90%',
    maxWidth: '500px',
    maxHeight: '90vh',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 20px',
    borderBottom: '1px solid #e5e7eb',
    flexShrink: 0,
  },
  title: {
    fontSize: '18px',
    fontWeight: '600',
    margin: 0,
  },
  closeButton: {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    color: '#6b7280',
    padding: '4px',
  },
  content: {
    flex: 1,
    overflow: 'auto',
    padding: '20px',
  },
  error: {
    backgroundColor: '#fef2f2',
    color: '#dc2626',
    padding: '12px',
    borderRadius: '8px',
    marginBottom: '16px',
    fontSize: '14px',
  },
  infoBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '12px',
    backgroundColor: '#f0fdf4',
    borderRadius: '8px',
    color: '#166534',
    fontSize: '14px',
    marginBottom: '16px',
  },
  description: {
    fontSize: '14px',
    color: '#6b7280',
    marginBottom: '20px',
    fontStyle: 'italic',
  },
  section: {
    marginBottom: '20px',
  },
  sectionTitle: {
    fontSize: '14px',
    fontWeight: '600',
    margin: '0 0 12px 0',
    color: '#374151',
  },
  variablesList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  variableField: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  label: {
    fontSize: '13px',
    fontWeight: '500',
    color: '#374151',
  },
  required: {
    color: '#dc2626',
    marginLeft: '4px',
  },
  input: {
    padding: '10px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
  },
  textarea: {
    padding: '10px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    minHeight: '80px',
    resize: 'vertical',
    fontFamily: 'inherit',
  },
  select: {
    padding: '10px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    backgroundColor: 'white',
  },
  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '12px',
    padding: '16px 20px',
    borderTop: '1px solid #e5e7eb',
    flexShrink: 0,
  },
  cancelButton: {
    padding: '10px 20px',
    border: '1px solid #d1d5db',
    borderRadius: '8px',
    backgroundColor: 'white',
    cursor: 'pointer',
    fontSize: '14px',
    color: '#374151',
  },
  executeButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 20px',
    border: 'none',
    borderRadius: '8px',
    backgroundColor: '#16a34a',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    color: 'white',
  },
};

export default AgentExecuteModal;
