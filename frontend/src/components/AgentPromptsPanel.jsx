import React, { useState, useEffect } from 'react';
import { Bot, Plus, Edit2, Trash2, Play, Settings } from 'lucide-react';
import { getAgentPrompts, deleteAgentPrompt } from '../services/api';
import AgentPromptModal from './AgentPromptModal';
import AgentExecuteModal from './AgentExecuteModal';

const ICON_MAP = {
  Bot: Bot,
  Settings: Settings,
};

const AgentPromptsPanel = ({
  notebookId,
  selectedDocuments = [],
  onAgentExecuted,
  onAgentExecuteStream
}) => {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingAgent, setEditingAgent] = useState(null);
  const [executeModalOpen, setExecuteModalOpen] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState(null);

  useEffect(() => {
    loadAgents();
  }, []);

  const loadAgents = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getAgentPrompts();
      setAgents(data);
    } catch (err) {
      console.error('Error loading agents:', err);
      setError('Error loading agents');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = () => {
    setEditingAgent(null);
    setModalOpen(true);
  };

  const handleEdit = (agent) => {
    setEditingAgent(agent);
    setModalOpen(true);
  };

  const handleDelete = async (agentId, agentName) => {
    if (!window.confirm(`Delete agent "${agentName}"?`)) return;
    try {
      await deleteAgentPrompt(agentId);
      setAgents(agents.filter(a => a.id !== agentId));
    } catch (err) {
      alert('Error during deletion');
    }
  };

  const handleExecute = (agent) => {
    if (selectedDocuments.length === 0) {
      alert('Select documents from the "Documents" tab first');
      return;
    }
    setSelectedAgent(agent);
    setExecuteModalOpen(true);
  };

  const handleSaveAgent = (savedAgent) => {
    if (editingAgent) {
      setAgents(agents.map(a => a.id === savedAgent.id ? savedAgent : a));
    } else {
      setAgents([...agents, savedAgent]);
    }
    setModalOpen(false);
  };

  const handleExecuteComplete = (conversationId) => {
    setExecuteModalOpen(false);
    if (onAgentExecuted) {
      onAgentExecuted(conversationId);
    }
  };

  const getIconComponent = (iconName) => {
    return ICON_MAP[iconName] || Bot;
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <button onClick={handleCreate} style={styles.createButton}>
          <Plus size={18} />
          New Agent
        </button>
      </div>

      {/* Info about document selection */}
      <div style={styles.infoBox}>
        {selectedDocuments.length === 0 ? (
          <span style={styles.infoTextWarning}>
            Select documents from the "Documents" tab to execute agents
          </span>
        ) : (
          <span style={styles.infoText}>
            {selectedDocuments.length} document(s) selected
          </span>
        )}
      </div>

      {/* Agent List */}
      <div style={styles.list}>
        {loading ? (
          <div style={styles.loading}>Loading...</div>
        ) : error ? (
          <div style={styles.error}>{error}</div>
        ) : agents.length === 0 ? (
          <div style={styles.empty}>
            <Bot size={48} style={{ opacity: 0.3 }} />
            <p style={styles.emptyTitle}>No agent configured</p>
            <p style={styles.emptySubtext}>
              Create an agent to automate document analysis
            </p>
          </div>
        ) : (
          agents.map(agent => {
            const IconComponent = getIconComponent(agent.icon);
            return (
              <div key={agent.id} style={styles.card}>
                <div style={styles.cardHeader}>
                  <div style={styles.cardIcon}>
                    <IconComponent size={20} />
                  </div>
                  <div style={styles.cardInfo}>
                    <h3 style={styles.cardTitle}>{agent.name}</h3>
                    {agent.description && (
                      <p style={styles.cardDescription}>{agent.description}</p>
                    )}
                    <div style={styles.cardMeta}>
                      <span>{agent.variables?.length || 0} variables</span>
                    </div>
                  </div>
                </div>

                <div style={styles.cardActions}>
                  <button
                    onClick={() => handleExecute(agent)}
                    style={{
                      ...styles.actionButton,
                      ...styles.executeButton,
                      ...(selectedDocuments.length === 0 ? styles.buttonDisabled : {})
                    }}
                    disabled={selectedDocuments.length === 0}
                    title={selectedDocuments.length === 0 ? 'Select documents' : 'Execute agent'}
                  >
                    <Play size={16} />
                  </button>
                  <button
                    onClick={() => handleEdit(agent)}
                    style={styles.actionButton}
                    title="Edit"
                  >
                    <Edit2 size={16} />
                  </button>
                  <button
                    onClick={() => handleDelete(agent.id, agent.name)}
                    style={{ ...styles.actionButton, ...styles.deleteButton }}
                    title="Delete"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Modals */}
      <AgentPromptModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSave={handleSaveAgent}
        editingAgent={editingAgent}
      />

      <AgentExecuteModal
        isOpen={executeModalOpen}
        onClose={() => setExecuteModalOpen(false)}
        agent={selectedAgent}
        selectedDocuments={selectedDocuments}
        notebookId={notebookId}
        onExecuteStream={(agent, documentIds, variables) => {
          setExecuteModalOpen(false);
          if (onAgentExecuteStream) {
            onAgentExecuteStream(agent, documentIds, variables);
          }
        }}
      />
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  },
  header: {
    padding: '12px 16px',
    borderBottom: '1px solid #e5e7eb',
    flexShrink: 0,
  },
  createButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    width: '100%',
    padding: '10px 16px',
    backgroundColor: '#2563eb',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    justifyContent: 'center',
  },
  infoBox: {
    padding: '8px 16px',
    backgroundColor: '#f9fafb',
    borderBottom: '1px solid #e5e7eb',
    flexShrink: 0,
  },
  infoText: {
    fontSize: '12px',
    color: '#059669',
  },
  infoTextWarning: {
    fontSize: '12px',
    color: '#d97706',
  },
  list: {
    flex: 1,
    overflow: 'auto',
    padding: '12px',
  },
  loading: {
    textAlign: 'center',
    padding: '40px 20px',
    color: '#6b7280',
  },
  error: {
    textAlign: 'center',
    padding: '40px 20px',
    color: '#dc2626',
  },
  empty: {
    textAlign: 'center',
    padding: '40px 20px',
    color: '#6b7280',
  },
  emptyTitle: {
    fontSize: '14px',
    fontWeight: '500',
    margin: '12px 0 4px',
  },
  emptySubtext: {
    fontSize: '12px',
    margin: 0,
    opacity: 0.7,
  },
  card: {
    backgroundColor: 'white',
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    padding: '12px',
    marginBottom: '8px',
  },
  cardHeader: {
    display: 'flex',
    gap: '12px',
    marginBottom: '8px',
  },
  cardIcon: {
    width: '36px',
    height: '36px',
    borderRadius: '8px',
    backgroundColor: '#f3f4f6',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#6b7280',
    flexShrink: 0,
  },
  cardInfo: {
    flex: 1,
    minWidth: 0,
  },
  cardTitle: {
    fontSize: '14px',
    fontWeight: '600',
    margin: 0,
    color: '#111827',
  },
  cardDescription: {
    fontSize: '12px',
    color: '#6b7280',
    margin: '4px 0 0',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  cardMeta: {
    fontSize: '11px',
    color: '#9ca3af',
    marginTop: '4px',
  },
  cardActions: {
    display: 'flex',
    gap: '8px',
    justifyContent: 'flex-end',
  },
  actionButton: {
    width: '32px',
    height: '32px',
    border: '1px solid #e5e7eb',
    borderRadius: '6px',
    backgroundColor: 'white',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#6b7280',
    transition: 'all 0.2s',
  },
  executeButton: {
    backgroundColor: '#dcfce7',
    borderColor: '#86efac',
    color: '#16a34a',
  },
  deleteButton: {
    color: '#dc2626',
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
};

export default AgentPromptsPanel;
