import React, { useState, useEffect } from 'react';
import { X, Plus, Trash2, Bot, Settings } from 'lucide-react';
import { createAgentPrompt, updateAgentPrompt } from '../services/api';

const ICONS = [
  { name: 'Bot', component: Bot },
  { name: 'Settings', component: Settings },
];

const DEFAULT_SYSTEM_PROMPT = `You are an AI assistant analyzing documents.

RULES:
1. Analyze ALL provided documents in the context
2. ALWAYS cite sources using the format: [document_name.pdf, page X]
3. Carefully follow user instructions`;

const AgentPromptModal = ({
  isOpen,
  onClose,
  onSave,
  editingAgent
}) => {
  // Basic info
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [icon, setIcon] = useState('Bot');

  // Prompts
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [userPrompt, setUserPrompt] = useState('');
  const [templatePrompt, setTemplatePrompt] = useState('');

  // Variables
  const [variables, setVariables] = useState([]);

  // UI state
  const [activeTab, setActiveTab] = useState('system');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  // Reset form when modal opens/closes or editingAgent changes
  useEffect(() => {
    if (isOpen) {
      if (editingAgent) {
        setName(editingAgent.name || '');
        setDescription(editingAgent.description || '');
        setIcon(editingAgent.icon || 'Bot');
        setSystemPrompt(editingAgent.system_prompt || DEFAULT_SYSTEM_PROMPT);
        setUserPrompt(editingAgent.user_prompt || '');
        setTemplatePrompt(editingAgent.template_prompt || '');
        setVariables(editingAgent.variables || []);
      } else {
        setName('');
        setDescription('');
        setIcon('Bot');
        setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
        setUserPrompt('');
        setTemplatePrompt('');
        setVariables([]);
      }
      setActiveTab('system');
      setError(null);
    }
  }, [isOpen, editingAgent]);

  const handleAddVariable = () => {
    setVariables([
      ...variables,
      {
        key: '',
        label: '',
        default: '',
        required: false,
        placeholder: ''
      }
    ]);
  };

  const handleRemoveVariable = (index) => {
    setVariables(variables.filter((_, i) => i !== index));
  };

  const handleVariableChange = (index, field, value) => {
    const updated = [...variables];
    updated[index] = { ...updated[index], [field]: value };
    setVariables(updated);
  };

  const handleSave = async () => {
    // Validation
    if (!name.trim()) {
      setError('Name is required');
      return;
    }
    if (!systemPrompt.trim()) {
      setError('System prompt is required');
      return;
    }
    if (!userPrompt.trim()) {
      setError('User prompt is required');
      return;
    }

    // Validate variables
    for (const v of variables) {
      if (!v.key.trim()) {
        setError('All variables must have a key');
        return;
      }
      if (!v.label.trim()) {
        setError('All variables must have a label');
        return;
      }
    }

    const data = {
      name: name.trim(),
      description: description.trim() || null,
      icon,
      system_prompt: systemPrompt,
      user_prompt: userPrompt,
      template_prompt: templatePrompt || null,
      variables: variables.map(v => ({
        key: v.key.trim(),
        label: v.label.trim(),
        default: v.default || null,
        required: v.required,
        placeholder: v.placeholder || null
      })),
    };

    try {
      setSaving(true);
      setError(null);

      let result;
      if (editingAgent) {
        result = await updateAgentPrompt(editingAgent.id, data);
      } else {
        result = await createAgentPrompt(data);
      }

      onSave(result);
    } catch (err) {
      setError('Error while saving');
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>
            {editingAgent ? 'Edit Agent' : 'New Agent'}
          </h2>
          <button onClick={onClose} style={styles.closeButton}>
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div style={styles.content}>
          {error && (
            <div style={styles.error}>{error}</div>
          )}

          {/* Basic Info */}
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Basic Information</h3>
            <div style={styles.row}>
              <div style={styles.field}>
                <label style={styles.label}>Name *</label>
                <input
                  type="text"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  style={styles.input}
                  placeholder="e.g. Technical Analysis"
                />
              </div>
              <div style={styles.fieldSmall}>
                <label style={styles.label}>Icon</label>
                <select
                  value={icon}
                  onChange={e => setIcon(e.target.value)}
                  style={styles.select}
                >
                  {ICONS.map(i => (
                    <option key={i.name} value={i.name}>{i.name}</option>
                  ))}
                </select>
              </div>
            </div>
            <div style={styles.field}>
              <label style={styles.label}>Description</label>
              <input
                type="text"
                value={description}
                onChange={e => setDescription(e.target.value)}
                style={styles.input}
                placeholder="Optional - short description of the agent"
              />
            </div>
          </div>

          {/* Prompt Tabs */}
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Prompt Configuration</h3>
            <div style={styles.tabs}>
              <button
                onClick={() => setActiveTab('system')}
                style={{
                  ...styles.tabButton,
                  ...(activeTab === 'system' ? styles.tabButtonActive : {})
                }}
              >
                System Prompt
              </button>
              <button
                onClick={() => setActiveTab('user')}
                style={{
                  ...styles.tabButton,
                  ...(activeTab === 'user' ? styles.tabButtonActive : {})
                }}
              >
                User Prompt *
              </button>
              <button
                onClick={() => setActiveTab('template')}
                style={{
                  ...styles.tabButton,
                  ...(activeTab === 'template' ? styles.tabButtonActive : {})
                }}
              >
                Template
              </button>
            </div>

            {activeTab === 'system' && (
              <div style={styles.tabContent}>
                <textarea
                  value={systemPrompt}
                  onChange={e => setSystemPrompt(e.target.value)}
                  style={styles.textarea}
                  placeholder="Define AI behavior and rules..."
                />
                <p style={styles.hint}>
                  Set AI behavior. Available variables: {`{{context}}`}
                  {variables.length > 0 && (
                    <span>
                      , {variables.map(v => `{{${v.key}}}`).join(', ')}
                    </span>
                  )}
                </p>
              </div>
            )}

            {activeTab === 'user' && (
              <div style={styles.tabContent}>
                <textarea
                  value={userPrompt}
                  onChange={e => setUserPrompt(e.target.value)}
                  style={styles.textarea}
                  placeholder="Write the main instruction using {{context}} for document content..."
                />
                <p style={styles.hint}>
                  Placeholder: {`{{context}}`} (documents)
                  {variables.length > 0 && (
                    <span>
                      , {variables.map(v => `{{${v.key}}}`).join(', ')}
                    </span>
                  )}
                </p>
              </div>
            )}

            {activeTab === 'template' && (
              <div style={styles.tabContent}>
                <textarea
                  value={templatePrompt}
                  onChange={e => setTemplatePrompt(e.target.value)}
                  style={styles.textarea}
                  placeholder="Optional - define output structure..."
                />
                <p style={styles.hint}>
                  Optional template to structure the output (e.g., markdown format)
                </p>
              </div>
            )}
          </div>

          {/* Variables */}
          <div style={styles.section}>
            <div style={styles.sectionHeader}>
              <h3 style={styles.sectionTitle}>Variabili Personalizzate</h3>
              <button onClick={handleAddVariable} style={styles.addButton}>
                <Plus size={14} /> Aggiungi
              </button>
            </div>

            {variables.length === 0 ? (
              <p style={styles.emptyText}>
                Nessuna variabile configurata. Aggiungi variabili per permettere configurazioni dinamiche.
              </p>
            ) : (
              <div style={styles.variablesList}>
                {variables.map((v, index) => (
                  <div key={index} style={styles.variableCard}>
                    <div style={styles.variableRow}>
                      <input
                        type="text"
                        value={v.key}
                        onChange={e => handleVariableChange(index, 'key', e.target.value)}
                        style={styles.inputSmall}
                        placeholder="chiave"
                      />
                      <input
                        type="text"
                        value={v.label}
                        onChange={e => handleVariableChange(index, 'label', e.target.value)}
                        style={styles.inputSmall}
                        placeholder="Etichetta"
                      />
                      <button
                        onClick={() => handleRemoveVariable(index)}
                        style={styles.removeButton}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    <div style={styles.variableRow}>
                      <input
                        type="text"
                        value={v.default || ''}
                        onChange={e => handleVariableChange(index, 'default', e.target.value)}
                        style={styles.inputSmall}
                        placeholder="Valore default"
                      />
                      <input
                        type="text"
                        value={v.placeholder || ''}
                        onChange={e => handleVariableChange(index, 'placeholder', e.target.value)}
                        style={styles.inputSmall}
                        placeholder="Placeholder"
                      />
                      <label style={styles.checkboxLabel}>
                        <input
                          type="checkbox"
                          checked={v.required}
                          onChange={e => handleVariableChange(index, 'required', e.target.checked)}
                        />
                        Obbligatoria
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={styles.footer}>
          <button onClick={onClose} style={styles.cancelButton} disabled={saving}>
            Annulla
          </button>
          <button onClick={handleSave} style={styles.saveButton} disabled={saving}>
            {saving ? 'Salvataggio...' : 'Salva Agent'}
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
    maxWidth: '700px',
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
  section: {
    marginBottom: '24px',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '12px',
  },
  sectionTitle: {
    fontSize: '14px',
    fontWeight: '600',
    margin: 0,
    color: '#374151',
  },
  row: {
    display: 'flex',
    gap: '12px',
    marginBottom: '12px',
  },
  field: {
    flex: 1,
  },
  fieldSmall: {
    width: '120px',
    flexShrink: 0,
  },
  label: {
    display: 'block',
    fontSize: '12px',
    fontWeight: '500',
    color: '#6b7280',
    marginBottom: '4px',
  },
  input: {
    width: '100%',
    padding: '8px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    boxSizing: 'border-box',
  },
  inputSmall: {
    flex: 1,
    padding: '6px 10px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '13px',
    minWidth: 0,
  },
  select: {
    width: '100%',
    padding: '8px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    backgroundColor: 'white',
    boxSizing: 'border-box',
  },
  tabs: {
    display: 'flex',
    gap: '4px',
    marginBottom: '12px',
    borderBottom: '1px solid #e5e7eb',
    paddingBottom: '8px',
  },
  tabButton: {
    padding: '8px 16px',
    border: 'none',
    backgroundColor: 'transparent',
    cursor: 'pointer',
    fontSize: '13px',
    color: '#6b7280',
    borderRadius: '6px',
    transition: 'all 0.2s',
  },
  tabButtonActive: {
    backgroundColor: '#eff6ff',
    color: '#2563eb',
    fontWeight: '500',
  },
  tabContent: {
    marginTop: '8px',
  },
  textarea: {
    width: '100%',
    height: '150px',
    padding: '12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '13px',
    fontFamily: 'monospace',
    resize: 'vertical',
    boxSizing: 'border-box',
  },
  hint: {
    fontSize: '11px',
    color: '#9ca3af',
    marginTop: '6px',
    marginBottom: 0,
  },
  addButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    padding: '6px 12px',
    backgroundColor: '#f3f4f6',
    border: '1px solid #e5e7eb',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '12px',
    color: '#374151',
  },
  emptyText: {
    fontSize: '13px',
    color: '#9ca3af',
    fontStyle: 'italic',
  },
  variablesList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  variableCard: {
    backgroundColor: '#f9fafb',
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    padding: '10px',
  },
  variableRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    marginBottom: '6px',
  },
  removeButton: {
    padding: '6px',
    backgroundColor: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: '#dc2626',
    flexShrink: 0,
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    fontSize: '12px',
    color: '#6b7280',
    whiteSpace: 'nowrap',
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
  saveButton: {
    padding: '10px 20px',
    border: 'none',
    borderRadius: '8px',
    backgroundColor: '#2563eb',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    color: 'white',
  },
};

export default AgentPromptModal;
