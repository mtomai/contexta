import { useState, useEffect } from 'react';
import { MessageSquare, Trash2, Plus } from 'lucide-react';
import { getConversations, deleteConversation, createConversation } from '../services/api';

/**
 * ConversationPanel component
 * Displays list of conversations with create/delete functionality
 */
function ConversationPanel({
  selectedConversationId,
  onSelectConversation,
  onConversationCreated,
  refreshTrigger,
  notebookId
}) {
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load conversations
  useEffect(() => {
    loadConversations();
  }, [refreshTrigger, notebookId]);

  const loadConversations = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getConversations(notebookId);
      setConversations(data);
    } catch (err) {
      setError('Error loading conversations');
      console.error('Error loading conversations:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateConversation = async () => {
    try {
      const newConversation = await createConversation('New Conversation', notebookId);
      setConversations([newConversation, ...conversations]);
      onSelectConversation(newConversation.id);
      if (onConversationCreated) {
        onConversationCreated(newConversation);
      }
    } catch (err) {
      console.error('Error creating conversation:', err);
      alert('Error creating conversation');
    }
  };

  const handleDeleteConversation = async (conversationId, e) => {
    e.stopPropagation(); // Prevent selecting conversation when deleting

    if (!window.confirm('Are you sure you want to delete this conversation?')) {
      return;
    }

    try {
      await deleteConversation(conversationId);
      setConversations(conversations.filter(c => c.id !== conversationId));

      // If deleted conversation was selected, clear selection
      if (selectedConversationId === conversationId) {
        onSelectConversation(null);
      }
    } catch (err) {
      console.error('Error deleting conversation:', err);
      alert('Error deleting conversation');
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

    return date.toLocaleDateString('en-US', { day: '2-digit', month: '2-digit' });
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      backgroundColor: '#f8f9fa'
    }}>
      {/* Header with New Conversation button */}
      <div style={{
        padding: '16px',
        borderBottom: '1px solid #e0e0e0',
        backgroundColor: 'white'
      }}>
        <button
          onClick={handleCreateConversation}
          style={{
            width: '100%',
            padding: '12px',
            backgroundColor: '#2563eb',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            transition: 'background-color 0.2s'
          }}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#1d4ed8'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#2563eb'}
        >
          <Plus size={18} />
          New Conversation
        </button>
      </div>

      {/* Conversations List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '8px'
      }}>
        {loading && (
          <div style={{
            padding: '20px',
            textAlign: 'center',
            color: '#6b7280'
          }}>
            Loading...
          </div>
        )}

        {error && (
          <div style={{
            padding: '20px',
            textAlign: 'center',
            color: '#ef4444'
          }}>
            {error}
          </div>
        )}

        {!loading && !error && conversations.length === 0 && (
          <div style={{
            padding: '20px',
            textAlign: 'center',
            color: '#6b7280',
            fontSize: '14px'
          }}>
            <MessageSquare size={32} style={{
              margin: '0 auto 12px',
              opacity: 0.5
            }} />
            <p>No conversations</p>
            <p style={{ fontSize: '12px', marginTop: '4px' }}>
              Click "New Conversation" to start
            </p>
          </div>
        )}

        {!loading && !error && conversations.map((conversation) => (
          <div
            key={conversation.id}
            onClick={() => onSelectConversation(conversation.id)}
            style={{
              padding: '12px',
              marginBottom: '8px',
              backgroundColor: selectedConversationId === conversation.id ? '#e0e7ff' : 'white',
              border: '1px solid',
              borderColor: selectedConversationId === conversation.id ? '#2563eb' : '#e0e0e0',
              borderRadius: '8px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              gap: '8px'
            }}
            onMouseEnter={(e) => {
              if (selectedConversationId !== conversation.id) {
                e.currentTarget.style.backgroundColor = '#f3f4f6';
              }
            }}
            onMouseLeave={(e) => {
              if (selectedConversationId !== conversation.id) {
                e.currentTarget.style.backgroundColor = 'white';
              }
            }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{
                fontSize: '14px',
                fontWeight: '600',
                color: '#1f2937',
                marginBottom: '4px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}>
                {conversation.title}
              </div>
              <div style={{
                fontSize: '12px',
                color: '#6b7280',
                display: 'flex',
                gap: '12px'
              }}>
                <span>{conversation.message_count} messages</span>
                <span>•</span>
                <span>{formatDate(conversation.updated_at)}</span>
              </div>
            </div>

            <button
              onClick={(e) => handleDeleteConversation(conversation.id, e)}
              style={{
                padding: '6px',
                backgroundColor: 'transparent',
                border: 'none',
                cursor: 'pointer',
                color: '#6b7280',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = '#fee2e2';
                e.target.style.color = '#ef4444';
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = 'transparent';
                e.target.style.color = '#6b7280';
              }}
            >
              <Trash2 size={16} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ConversationPanel;
