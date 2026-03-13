import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
  },
});

/**
 * Upload a document (PDF or Word)
 * @param {File} file - The file to upload
 * @param {string} notebookId - Optional notebook ID to associate document with
 * @param {Function} onProgress - Progress callback (optional)
 * @returns {Promise} Document upload response
 */
export const uploadDocument = async (file, notebookId = null, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);
  if (notebookId) {
    formData.append('notebook_id', notebookId);
  }

  const response = await api.post('/api/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress(percentCompleted);
      }
    },
  });

  return response.data;
};

/**
 * Get list of all documents
 * @param {string} notebookId - Optional notebook ID to filter documents
 * @returns {Promise} List of documents
 */
export const getDocuments = async (notebookId = null) => {
  const url = notebookId
    ? `/api/notebooks/${notebookId}/documents`
    : '/api/documents';
  const response = await api.get(url, {
    params: { _t: Date.now() },
  });
  return response.data;
};

/**
 * Delete a document
 * @param {string} documentId - Document ID to delete
 * @returns {Promise} Deletion response
 */
export const deleteDocument = async (documentId) => {
  const response = await api.delete(`/api/documents/${documentId}`);
  return response.data;
};

/**
 * Send a chat message
 * @param {string} query - User query
 * @param {string} conversationId - Optional conversation ID
 * @returns {Promise} Chat response with answer, sources, and conversation_id
 */
export const sendChatMessage = async (query, conversationId = null) => {
  const payload = { query };
  if (conversationId) {
    payload.conversation_id = conversationId;
  }
  const response = await api.post('/api/chat', payload);
  return response.data;
};

/**
 * Send a chat message with streaming response
 * @param {string} query - User query
 * @param {string} conversationId - Optional conversation ID
 * @param {Object} callbacks - Callback functions for streaming events
 * @param {Function} callbacks.onSources - Called when sources are received
 * @param {Function} callbacks.onToken - Called for each token received
 * @param {Function} callbacks.onDone - Called when streaming is complete
 * @param {Function} callbacks.onError - Called on error
 * @returns {Promise} AbortController to cancel the request
 */
export const sendChatMessageStreaming = async (query, conversationId = null, callbacks = {}) => {
  const { onSources, onToken, onDone, onError, onThought } = callbacks;
  const abortController = new AbortController();
  
  const payload = { query };
  if (conversationId) {
    payload.conversation_id = conversationId;
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: abortController.signal,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete SSE events
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer
      
      let currentEvent = null;
      
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ') && currentEvent) {
          const data = JSON.parse(line.slice(6));
          
          switch (currentEvent) {
            case 'sources':
              if (onSources) onSources(data.sources);
              break;
            case 'thought':
              if (onThought) onThought(data.message);
              break;
            case 'token':
              if (onToken) onToken(data.token);
              break;
            case 'done':
              if (onDone) onDone(data);
              break;
            case 'error':
              if (onError) onError(data.message);
              break;
          }
          
          currentEvent = null;
        }
      }
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('Streaming request aborted');
    } else {
      if (onError) onError(error.message);
      throw error;
    }
  }
  
  return abortController;
};

/**
 * Get embedding cache statistics
 * @returns {Promise} Cache statistics
 */
export const getCacheStats = async () => {
  const response = await api.get('/api/chat/cache/stats');
  return response.data;
};

/**
 * Clear the embedding cache
 * @returns {Promise} Clear response
 */
export const clearCache = async () => {
  const response = await api.post('/api/chat/cache/clear');
  return response.data;
};

/**
 * Get document statistics
 * @returns {Promise} Statistics
 */
export const getStats = async () => {
  const response = await api.get('/api/documents/stats');
  return response.data;
};

// ===== CONVERSATION API CALLS =====

/**
 * Get all conversations
 * @param {string} notebookId - Optional notebook ID to filter conversations
 * @returns {Promise} List of conversations
 */
export const getConversations = async (notebookId = null) => {
  const url = notebookId
    ? `/api/notebooks/${notebookId}/conversations`
    : '/api/conversations';
  const response = await api.get(url);
  return response.data;
};

/**
 * Create a new conversation
 * @param {string} title - Optional conversation title
 * @param {string} notebookId - Optional notebook ID to associate with
 * @returns {Promise} Created conversation
 */
export const createConversation = async (title = 'Nuova Conversazione', notebookId = null) => {
  const payload = { title };
  if (notebookId) {
    payload.notebook_id = notebookId;
  }
  const response = await api.post('/api/conversations', payload);
  return response.data;
};

/**
 * Get a specific conversation with messages
 * @param {string} conversationId - Conversation ID
 * @returns {Promise} Conversation with messages
 */
export const getConversation = async (conversationId) => {
  const response = await api.get(`/api/conversations/${conversationId}`);
  return response.data;
};

/**
 * Delete a conversation
 * @param {string} conversationId - Conversation ID to delete
 * @returns {Promise} Deletion response
 */
export const deleteConversation = async (conversationId) => {
  const response = await api.delete(`/api/conversations/${conversationId}`);
  return response.data;
};

/**
 * Update conversation title
 * @param {string} conversationId - Conversation ID
 * @param {string} newTitle - New title
 * @returns {Promise} Updated conversation
 */
export const updateConversationTitle = async (conversationId, newTitle) => {
  const response = await api.put(`/api/conversations/${conversationId}`, {
    title: newTitle
  });
  return response.data;
};

// ===== NOTEBOOK API CALLS =====

/**
 * Get all notebooks
 * @returns {Promise} List of notebooks
 */
export const getNotebooks = async () => {
  const response = await api.get('/api/notebooks');
  return response.data;
};

/**
 * Create a new notebook
 * @param {string} name - Notebook name
 * @param {string} description - Optional description
 * @returns {Promise} Created notebook
 */
export const createNotebook = async (name, description = null) => {
  const response = await api.post('/api/notebooks', { name, description });
  return response.data;
};

/**
 * Get a specific notebook with statistics
 * @param {string} notebookId - Notebook ID
 * @returns {Promise} Notebook with stats
 */
export const getNotebook = async (notebookId) => {
  const response = await api.get(`/api/notebooks/${notebookId}`);
  return response.data;
};

/**
 * Update notebook
 * @param {string} notebookId - Notebook ID
 * @param {string} name - New name
 * @param {string} description - New description
 * @returns {Promise} Updated notebook
 */
export const updateNotebook = async (notebookId, name, description = null) => {
  const response = await api.put(`/api/notebooks/${notebookId}`, {
    name,
    description
  });
  return response.data;
};

/**
 * Delete a notebook
 * @param {string} notebookId - Notebook ID to delete
 * @returns {Promise} Deletion response
 */
export const deleteNotebook = async (notebookId) => {
  const response = await api.delete(`/api/notebooks/${notebookId}`);
  return response.data;
};

// ===== AGENT PROMPTS API CALLS =====

/**
 * Get all agent prompts for a notebook
 * @param {string} notebookId - Notebook ID
 * @returns {Promise} List of agent prompts
 */
export const getAgentPrompts = async () => {
  const response = await api.get(`/api/agent-prompts`);
  return response.data;
};

/**
 * Get a single agent prompt by ID
 * @param {string} promptId - Agent prompt ID
 * @returns {Promise} Agent prompt
 */
export const getAgentPrompt = async (promptId) => {
  const response = await api.get(`/api/agent-prompts/${promptId}`);
  return response.data;
};

/**
 * Create a new agent prompt
 * @param {string} notebookId - Notebook ID
 * @param {Object} data - Agent prompt data
 * @returns {Promise} Created agent prompt
 */
export const createAgentPrompt = async (data) => {
  const response = await api.post(`/api/agent-prompts`, data);
  return response.data;
};

/**
 * Update an agent prompt
 * @param {string} promptId - Agent prompt ID
 * @param {Object} data - Update data
 * @returns {Promise} Updated agent prompt
 */
export const updateAgentPrompt = async (promptId, data) => {
  const response = await api.put(`/api/agent-prompts/${promptId}`, data);
  return response.data;
};

/**
 * Delete an agent prompt
 * @param {string} promptId - Agent prompt ID to delete
 * @returns {Promise} Deletion response
 */
export const deleteAgentPrompt = async (promptId) => {
  const response = await api.delete(`/api/agent-prompts/${promptId}`);
  return response.data;
};

/**
 * Execute an agent prompt on selected documents
 * @param {string} promptId - Agent prompt ID to execute
 * @param {Array} documentIds - List of document IDs
 * @param {string} notebookId - Notebook ID
 * @param {Object} variableValues - Values for dynamic variables
 * @param {Object} options - Execution options override
 * @returns {Promise} Execution response with conversation_id
 */
export const executeAgentPrompt = async (promptId, documentIds, notebookId, variableValues = {}) => {
  const response = await api.post(`/api/agent-prompts/${promptId}/execute`, {
    document_ids: documentIds,
    notebook_id: notebookId,
    variable_values: variableValues
  });
  return response.data;
};

/**
 * Execute an agent prompt with streaming SSE response
 * @param {string} promptId - Agent prompt ID
 * @param {Object} data - Payload with document_ids, notebook_id, variable_values
 * @param {Function} onThought - Called when a thought event is received
 * @param {Function} onToken - Called for each token received
 * @param {Function} onDone - Called when streaming is complete (receives {conversation_id, title, sources})
 * @param {Function} onError - Called on error
 * @returns {Promise<AbortController>} AbortController to cancel the request
 */
export const executeAgentStream = async (promptId, data, onThought, onToken, onDone, onError) => {
  const abortController = new AbortController();

  try {
    const response = await fetch(`${API_BASE_URL}/api/agent-prompts/${promptId}/execute_stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      signal: abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let currentEvent = null;

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ') && currentEvent) {
          const eventData = JSON.parse(line.slice(6));

          switch (currentEvent) {
            case 'thought':
              if (onThought) onThought(eventData.message);
              break;
            case 'token':
              if (onToken) onToken(eventData.content);
              break;
            case 'done':
              if (onDone) onDone(eventData);
              break;
            case 'error':
              if (onError) onError(eventData.message);
              break;
          }

          currentEvent = null;
        }
      }
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('Agent stream request aborted');
    } else {
      if (onError) onError(error.message);
    }
  }

  return abortController;
};

// ===== NOTES API CALLS =====

/**
 * Get all saved notes for a notebook
 * @param {string} notebookId - Notebook ID
 * @returns {Promise} List of notes
 */
export const getNotes = async (notebookId) => {
  const response = await api.get(`/api/notebooks/${notebookId}/notes`);
  return response.data;
};

/**
 * Create a new saved note (pin an AI response)
 * @param {string} notebookId - Notebook ID
 * @param {string} content - Note content
 * @returns {Promise} Created note
 */
export const createNote = async (notebookId, content) => {
  const response = await api.post(`/api/notebooks/${notebookId}/notes`, { content });
  return response.data;
};

/**
 * Delete a saved note
 * @param {string} noteId - Note ID to delete
 * @returns {Promise} Deletion response
 */
export const deleteNote = async (noteId) => {
  const response = await api.delete(`/api/notes/${noteId}`);
  return response.data;
};

export default api;
