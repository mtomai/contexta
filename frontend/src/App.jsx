import React, { useState } from 'react';
import DocumentUploader from './components/DocumentUploader';
import DocumentList from './components/DocumentList';
import ChatInterface from './components/ChatInterface';
import SourceViewer from './components/SourceViewer';
import ConversationPanel from './components/ConversationPanel';
import NotebookPanel from './components/NotebookPanel';
import NotebookHeader from './components/NotebookHeader';
import AgentPromptsPanel from './components/AgentPromptsPanel';
import NotesPanel from './components/NotesPanel';

function App() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [selectedSource, setSelectedSource] = useState(null);
  const [sidebarTab, setSidebarTab] = useState('conversations'); // 'documents' or 'conversations'
  const [selectedConversationId, setSelectedConversationId] = useState(null);
  const [conversationRefreshTrigger, setConversationRefreshTrigger] = useState(0);

  // Notebook state
  const [selectedNotebookId, setSelectedNotebookId] = useState(null);
  const [notebookRefreshTrigger, setNotebookRefreshTrigger] = useState(0);

  // Selected documents for custom prompts
  const [selectedDocuments, setSelectedDocuments] = useState([]);

  // Notes refresh trigger
  const [notesRefreshTrigger, setNotesRefreshTrigger] = useState(0);

  // Agent stream trigger - passed to ChatInterface to start streaming
  const [agentStreamRequest, setAgentStreamRequest] = useState(null);

  const handleUploadSuccess = () => {
    // Trigger refresh of document list
    setRefreshTrigger(prev => prev + 1);
  };

  const handleSourceClick = (source) => {
    setSelectedSource(source);
  };

  const handleCloseSource = () => {
    setSelectedSource(null);
  };

  const handleSelectConversation = (conversationId) => {
    setSelectedConversationId(conversationId);
  };

  const handleConversationCreated = () => {
    // Refresh conversation list
    setConversationRefreshTrigger(prev => prev + 1);
  };

  const handleMessageSent = () => {
    // Refresh conversation list to update message count and timestamp
    setConversationRefreshTrigger(prev => prev + 1);
  };

  const handleSelectNotebook = (notebookId) => {
    setSelectedNotebookId(notebookId);
    setSelectedConversationId(null); // Reset selected conversation
    setSelectedDocuments([]); // Reset selected documents
    setSidebarTab('conversations'); // Default to conversations tab
  };

  const handleBackToNotebooks = () => {
    setSelectedNotebookId(null);
    setSelectedConversationId(null);
    setSelectedDocuments([]); // Reset selected documents
  };

  const handleDocumentSelectionChange = (documentIds) => {
    setSelectedDocuments(documentIds);
  };

  const handleAgentExecuted = (conversationId) => {
    // Navigate to the new conversation
    setSelectedConversationId(conversationId);
    setSidebarTab('conversations');
    setConversationRefreshTrigger(prev => prev + 1);
  };

  const handleAgentExecuteStream = (agent, documentIds, variableValues) => {
    // Clear any selected conversation so ChatInterface enters "agent streaming" mode
    setSelectedConversationId(null);
    // Switch to conversations tab to show the chat
    setSidebarTab('conversations');
    // Pass agent stream request to ChatInterface
    setAgentStreamRequest({
      agent,
      documentIds,
      notebookId: selectedNotebookId,
      variableValues,
      timestamp: Date.now() // ensures uniqueness for useEffect trigger
    });
  };

  const handleNotePinned = () => {
    setNotesRefreshTrigger(prev => prev + 1);
  };

  // If no notebook selected, show notebook selection view
  if (!selectedNotebookId) {
    return (
      <div style={styles.app}>
        <NotebookPanel
          selectedNotebookId={selectedNotebookId}
          onSelectNotebook={handleSelectNotebook}
          refreshTrigger={notebookRefreshTrigger}
        />
        {selectedSource && (
          <SourceViewer source={selectedSource} onClose={handleCloseSource} />
        )}
      </div>
    );
  }

  // Normal view with selected notebook
  return (
    <div style={styles.app}>
      <div style={styles.sidebar}>
        {/* Notebook Header */}
        <NotebookHeader
          notebookId={selectedNotebookId}
          onBack={handleBackToNotebooks}
        />

        {/* Tab Switcher */}
        <div style={styles.tabContainer}>
          <button
            onClick={() => setSidebarTab('conversations')}
            style={{
              ...styles.tab,
              ...(sidebarTab === 'conversations' ? styles.tabActive : {})
            }}
          >
            Conversations
          </button>
          <button
            onClick={() => setSidebarTab('documents')}
            style={{
              ...styles.tab,
              ...(sidebarTab === 'documents' ? styles.tabActive : {})
            }}
          >
            Documents
          </button>
          <button
            onClick={() => setSidebarTab('agents')}
            style={{
              ...styles.tab,
              ...(sidebarTab === 'agents' ? styles.tabActive : {})
            }}
          >
            Agents
          </button>
          <button
            onClick={() => setSidebarTab('notes')}
            style={{
              ...styles.tab,
              ...(sidebarTab === 'notes' ? styles.tabActive : {})
            }}
          >
            Notes
          </button>
        </div>

        {/* Conditional Content */}
        {sidebarTab === 'documents' && (
          <div style={styles.tabContent}>
            <DocumentUploader
              onUploadSuccess={handleUploadSuccess}
              notebookId={selectedNotebookId}
            />
            <DocumentList
              refreshTrigger={refreshTrigger}
              notebookId={selectedNotebookId}
              selectedDocuments={selectedDocuments}
              onSelectionChange={handleDocumentSelectionChange}
            />
          </div>
        )}

        {sidebarTab === 'conversations' && (
          <div style={styles.tabContent}>
            <ConversationPanel
              selectedConversationId={selectedConversationId}
              onSelectConversation={handleSelectConversation}
              onConversationCreated={handleConversationCreated}
              refreshTrigger={conversationRefreshTrigger}
              notebookId={selectedNotebookId}
            />
          </div>
        )}

        {sidebarTab === 'agents' && (
          <div style={styles.tabContent}>
            <AgentPromptsPanel
              notebookId={selectedNotebookId}
              selectedDocuments={selectedDocuments}
              onAgentExecuted={handleAgentExecuted}
              onAgentExecuteStream={handleAgentExecuteStream}
            />
          </div>
        )}

        {sidebarTab === 'notes' && (
          <div style={styles.tabContent}>
            <NotesPanel
              notebookId={selectedNotebookId}
              refreshTrigger={notesRefreshTrigger}
            />
          </div>
        )}
      </div>

      <div style={styles.main}>
        <ChatInterface
          onSourceClick={handleSourceClick}
          selectedConversationId={selectedConversationId}
          onMessageSent={handleMessageSent}
          notebookId={selectedNotebookId}
          selectedDocuments={selectedDocuments}
          onNotePinned={handleNotePinned}
          agentStreamRequest={agentStreamRequest}
          onAgentStreamComplete={(conversationId) => {
            setSelectedConversationId(conversationId);
            setConversationRefreshTrigger(prev => prev + 1);
            setAgentStreamRequest(null);
          }}
        />
      </div>

      {selectedSource && (
        <SourceViewer source={selectedSource} onClose={handleCloseSource} />
      )}
    </div>
  );
}

const styles = {
  app: {
    display: 'flex',
    height: '100vh',
    overflow: 'hidden',
  },
  sidebar: {
    width: '350px',
    backgroundColor: '#fafafa',
    borderRight: '1px solid #e0e0e0',
    display: 'flex',
    flexDirection: 'column',
    flexShrink: 0,
  },
  tabContainer: {
    display: 'flex',
    backgroundColor: 'white',
    borderBottom: '1px solid #e0e0e0',
    flexShrink: 0,
  },
  tab: {
    flex: 1,
    padding: '12px',
    border: 'none',
    backgroundColor: 'transparent',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    color: '#6b7280',
    borderBottom: '2px solid transparent',
    transition: 'all 0.2s',
  },
  tabActive: {
    color: '#2563eb',
    borderBottomColor: '#2563eb',
    fontWeight: '600',
  },
  tabContent: {
    flex: 1,
    overflow: 'hidden',
    minHeight: 0,
    display: 'flex',
    flexDirection: 'column',
  },
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
};

export default App;
