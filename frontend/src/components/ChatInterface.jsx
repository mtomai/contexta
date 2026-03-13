import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader, MessageSquare, Zap } from 'lucide-react';
import { sendChatMessage, sendChatMessageStreaming, getConversation, createNote, executeAgentStream } from '../services/api';
import MessageBubble from './MessageBubble';

const ChatInterface = ({ onSourceClick, selectedConversationId, onMessageSent, notebookId, selectedDocuments = [], onNotePinned, agentStreamRequest, onAgentStreamComplete }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [streamingContent, setStreamingContent] = useState('');
  const [isAgentStreaming, setIsAgentStreaming] = useState(false);

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load conversation messages when selectedConversationId changes
  useEffect(() => {
    const loadConversationMessages = async () => {
      if (!selectedConversationId) {
        // Don't clear messages if we're in agent streaming mode
        if (!isAgentStreaming) {
          setMessages([]);
        }
        return;
      }

      try {
        setLoadingConversation(true);
        const data = await getConversation(selectedConversationId);

        // Transform backend messages to frontend format
        const transformedMessages = data.messages.map(msg => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          sources: msg.sources || [],
          timestamp: new Date(msg.timestamp),
          isError: msg.is_error
        }));

        setMessages(transformedMessages);
      } catch (err) {
        console.error('Error loading conversation:', err);
        setMessages([]);
      } finally {
        setLoadingConversation(false);
      }
    };

    loadConversationMessages();
  }, [selectedConversationId]);

  // Handle agent streaming request from AgentPromptsPanel
  useEffect(() => {
    if (!agentStreamRequest) return;

    const { agent, documentIds, notebookId: agentNotebookId, variableValues } = agentStreamRequest;

    // Build display text for variables
    const varsDisplay = variableValues && Object.keys(variableValues).length > 0
      ? Object.entries(variableValues).map(([k, v]) => `${k}=${v}`).join(', ')
      : 'none';

    const userMessageId = Date.now();
    const assistantMessageId = Date.now() + 1;

    // Insert user message and empty assistant message
    setMessages([
      {
        id: userMessageId,
        role: 'user',
        content: `[Agent: ${agent.name}]\nVariables: ${varsDisplay}`,
        timestamp: new Date()
      },
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        sources: [],
        thoughts: [],
        timestamp: new Date(),
        isStreaming: true
      }
    ]);

    setIsAgentStreaming(true);
    setLoading(true);
    setStreamingContent('');
    let fullContent = '';

    const payload = {
      document_ids: documentIds,
      notebook_id: agentNotebookId,
      variable_values: variableValues
    };

    executeAgentStream(
      agent.id,
      payload,
      // onThought
      (thoughtMessage) => {
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId
            ? { ...msg, thoughts: [...(msg.thoughts || []), thoughtMessage] }
            : msg
        ));
      },
      // onToken
      (token) => {
        fullContent += token;
        setStreamingContent(fullContent);
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId
            ? { ...msg, content: fullContent }
            : msg
        ));
      },
      // onDone
      (data) => {
        const finalSources = data.sources || [];
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: fullContent,
                sources: finalSources,
                isStreaming: false,
                thoughts: msg.thoughts || []
              }
            : msg
        ));
        setLoading(false);
        setIsAgentStreaming(false);
        // Navigate to the created conversation
        if (onAgentStreamComplete && data.conversation_id) {
          onAgentStreamComplete(data.conversation_id);
        }
      },
      // onError
      (errorMsg) => {
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId
            ? { ...msg, content: `Error: ${errorMsg}`, isError: true, isStreaming: false }
            : msg
        ));
        setLoading(false);
        setIsAgentStreaming(false);
      }
    );
  }, [agentStreamRequest]);



  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!inputValue.trim() || loading) {
      return;
    }

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);
    setStreamingContent('');

    if (streamingEnabled) {
      // Streaming mode
      const assistantMessageId = Date.now() + 1;
      let fullContent = '';

      // Add empty assistant message that will be updated
      setMessages(prev => [...prev, {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        sources: [],
        thoughts: [],
        timestamp: new Date(),
        isStreaming: true
      }]);

      try {
        abortControllerRef.current = await sendChatMessageStreaming(
          userMessage.content,
          selectedConversationId,
          {
            onThought: (thoughtMessage) => {
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMessageId
                  ? { ...msg, thoughts: [...(msg.thoughts || []), thoughtMessage] }
                  : msg
              ));
            },
            onToken: (token) => {
              fullContent += token;
              setStreamingContent(fullContent);
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMessageId
                  ? { ...msg, content: fullContent }
                  : msg
              ));
            },
            onDone: (data) => {
              // Sources are now sent with the done event, after filtering
              const filteredSources = data.sources || [];
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMessageId
                  ? {
                      ...msg,
                      content: data.full_response || fullContent,
                      sources: filteredSources,
                      isStreaming: false,
                      thoughts: msg.thoughts || []
                    }
                  : msg
              ));
              setLoading(false);
              if (onMessageSent) onMessageSent();
            },
            onError: (errorMsg) => {
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMessageId
                  ? { ...msg, content: `Error: ${errorMsg}`, isError: true, isStreaming: false }
                  : msg
              ));
              setLoading(false);
            }
          }
        );
      } catch (err) {
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId
            ? { ...msg, content: 'Error processing request.', isError: true, isStreaming: false }
            : msg
        ));
        setLoading(false);
        console.error(err);
      }
    } else {
      // Non-streaming mode (original behavior)
      try {
        const response = await sendChatMessage(userMessage.content, selectedConversationId);

        const assistantMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, assistantMessage]);

        if (onMessageSent) {
          onMessageSent();
        }

      } catch (err) {
        const errorMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: 'Error processing request. Please try again later.',
          isError: true,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, errorMessage]);
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.headerTop}>
          <h1 style={styles.title}>Contexta</h1>
          <div style={styles.headerActions}>
            {/* Streaming toggle */}
            <button
              onClick={() => setStreamingEnabled(!streamingEnabled)}
              style={{
                ...styles.streamingToggle,
                ...(streamingEnabled ? styles.streamingToggleActive : {})
              }}
              title={streamingEnabled ? 'Streaming attivo' : 'Streaming disattivo'}
            >
              <Zap size={16} />
              <span>{streamingEnabled ? 'Stream ON' : 'Stream OFF'}</span>
            </button>
          </div>
        </div>
        <p style={styles.subtitle}>
          {isAgentStreaming
            ? 'Esecuzione agent in corso...'
            : selectedConversationId
              ? 'Fai domande sui tuoi documenti caricati'
              : 'Seleziona o crea una conversazione per iniziare'}
          {selectedDocuments.length > 0 && (
            <span style={styles.selectedCount}>
              {' '}• {selectedDocuments.length} doc selezionati
            </span>
          )}
        </p>
      </div>

      <div style={styles.messagesContainer}>
        {!selectedConversationId && !isAgentStreaming ? (
          <div style={styles.emptyState}>
            <MessageSquare size={64} style={{ color: '#ccc', marginBottom: '20px' }} />
            <p style={styles.emptyText}>
              Nessuna conversazione selezionata
            </p>
            <p style={styles.emptySubtext}>
              Seleziona una conversazione esistente o creane una nuova per iniziare a chattare
            </p>
          </div>
        ) : loadingConversation ? (
          <div style={styles.emptyState}>
            <Loader size={32} style={styles.spinner} />
            <p style={styles.emptySubtext}>Caricamento conversazione...</p>
          </div>
        ) : messages.length === 0 ? (
          <div style={styles.emptyState}>
            <p style={styles.emptyText}>
              Carica alcuni documenti e inizia a fare domande!
            </p>
            <p style={styles.emptySubtext}>
              Esempi di domande:
            </p>
            <ul style={styles.examplesList}>
              <li>"Di cosa parla questo documento?"</li>
              <li>"Quali sono i punti principali?"</li>
              <li>"Riassumi il contenuto"</li>
            </ul>
          </div>
        ) : (
          <div style={styles.messagesList}>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onSourceClick={onSourceClick}
                onPinNote={notebookId ? async (content) => {
                  await createNote(notebookId, content);
                  if (onNotePinned) onNotePinned();
                } : undefined}
              />
            ))}
            {loading && !streamingEnabled && (
              <div style={styles.loadingIndicator}>
                <Loader size={20} style={styles.spinner} />
                <span>Sto pensando...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} style={styles.inputForm} data-chat-form>
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            selectedConversationId
              ? "Fai una domanda sui tuoi documenti..."
              : "Seleziona una conversazione per iniziare..."
          }
          style={styles.textarea}
          rows={3}
          disabled={loading || !selectedConversationId}
        />
        <button
          type="submit"
          style={{
            ...styles.sendButton,
            ...(loading || !inputValue.trim() || !selectedConversationId ? styles.sendButtonDisabled : {})
          }}
          disabled={loading || !inputValue.trim() || !selectedConversationId}
        >
          <Send size={20} />
        </button>
      </form>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    backgroundColor: '#fff',
  },
  header: {
    padding: '24px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
  },
  headerTop: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    flexWrap: 'wrap',
  },
  selectedCount: {
    color: '#2563eb',
    fontWeight: '500',
  },
  title: {
    fontSize: '28px',
    fontWeight: '700',
    color: '#333',
    margin: 0,
  },
  streamingToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '8px 12px',
    border: '1px solid #ddd',
    borderRadius: '20px',
    backgroundColor: '#fff',
    color: '#666',
    fontSize: '12px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  streamingToggleActive: {
    backgroundColor: '#e8f5e9',
    borderColor: '#4caf50',
    color: '#2e7d32',
  },
  subtitle: {
    fontSize: '16px',
    color: '#666',
    margin: 0,
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
  },
  emptyState: {
    textAlign: 'center',
    padding: '60px 20px',
    color: '#666',
  },
  emptyText: {
    fontSize: '18px',
    marginBottom: '24px',
  },
  emptySubtext: {
    fontSize: '16px',
    marginBottom: '12px',
    fontWeight: '500',
  },
  examplesList: {
    textAlign: 'left',
    display: 'inline-block',
    listStyle: 'none',
    padding: 0,
  },

  messagesList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  loadingIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '12px 16px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    width: 'fit-content',
    color: '#666',
  },
  spinner: {
    animation: 'spin 1s linear infinite',
  },
  inputForm: {
    display: 'flex',
    gap: '12px',
    padding: '20px',
    borderTop: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
  },
  textarea: {
    flex: 1,
    padding: '12px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    fontSize: '14px',
    fontFamily: 'inherit',
    resize: 'none',
    outline: 'none',
  },
  sendButton: {
    padding: '12px 16px',
    backgroundColor: '#4a90e2',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed',
  },
};

export default ChatInterface;
