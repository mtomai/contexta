import React, { useMemo, useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot, FileText, Loader, Pin, Check, CheckCircle } from 'lucide-react';

// Parse citations [documento, pagina X] and make them clickable
const parseCitations = (content, sources, onSourceClick) => {
  if (!content) return null;

  // Pattern to match citations like [document.pdf, page 5] or [document, p. 5]
  const citationRegex = /\[([^\],]+),\s*(?:page|p\.?)\s*(\d+)\]/gi;

  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = citationRegex.exec(content)) !== null) {
    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push({
        type: 'text',
        content: content.slice(lastIndex, match.index)
      });
    }

    // Add the citation
    const docName = match[1].trim();
    const pageNum = parseInt(match[2], 10);

    // Find matching source
    const matchingSource = sources?.find(
      s => s.document === docName && s.page === pageNum
    );

    parts.push({
      type: 'citation',
      docName,
      pageNum,
      fullMatch: match[0],
      source: matchingSource
    });

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < content.length) {
    parts.push({
      type: 'text',
      content: content.slice(lastIndex)
    });
  }

  // If no citations found, return null to use default rendering
  if (parts.length === 1 && parts[0].type === 'text') {
    return null;
  }

  return parts.map((part, index) => {
    if (part.type === 'text') {
      return <span key={index}>{part.content}</span>;
    } else {
      return (
        <span
          key={index}
          style={{
            ...styles.citationLink,
            ...(part.source ? styles.citationLinkActive : styles.citationLinkInactive)
          }}
          onClick={(e) => {
            e.stopPropagation();
            if (part.source && onSourceClick) {
              onSourceClick(part.source);
            }
          }}
          title={part.source ? `Click to view source` : `Source not found in results`}
        >
          {part.fullMatch}
        </span>
      );
    }
  });
};

const MessageBubble = ({ message, onSourceClick, onPinNote }) => {
  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming;
  const [pinned, setPinned] = useState(false);
  const [pinning, setPinning] = useState(false);
  const [isThoughtsOpen, setIsThoughtsOpen] = useState(!message.content);

  useEffect(() => {
    if (message.content && message.content.length > 0) {
      setIsThoughtsOpen(false);
    }
  }, [message.content]);

  const handlePin = async () => {
    if (pinned || pinning || !onPinNote) return;
    setPinning(true);
    try {
      await onPinNote(message.content);
      setPinned(true);
    } catch (err) {
      console.error('Error pinning note:', err);
    } finally {
      setPinning(false);
    }
  };

  // Custom text renderer that makes citations clickable
  const renderTextWithCitations = useMemo(() => {
    return ({ children }) => {
      if (typeof children === 'string') {
        const parsed = parseCitations(children, message.sources, onSourceClick);
        if (parsed) {
          return <>{parsed}</>;
        }
      }
      return <>{children}</>;
    };
  }, [message.sources, onSourceClick]);

  return (
    <div style={{
      ...styles.container,
      justifyContent: isUser ? 'flex-end' : 'flex-start'
    }}>
      {!isUser && (
        <div style={styles.avatarBot}>
          <Bot size={20} color="#fff" />
        </div>
      )}

      <div style={{
        ...styles.bubble,
        ...(isUser ? styles.bubbleUser : styles.bubbleAssistant),
        ...(message.isError ? styles.bubbleError : {}),
        ...(isStreaming ? styles.bubbleStreaming : {})
      }}>
        {isUser ? (
          <p style={styles.messageText}>{message.content}</p>
        ) : (
          <div>
            {/* Chain of Thought — accordion */}
            {message.thoughts && message.thoughts.length > 0 && (
              <div className="thoughts-container">
                <button
                  className="thoughts-summary"
                  onClick={() => setIsThoughtsOpen(!isThoughtsOpen)}
                >
                  <span className="thoughts-summary-icon">
                    {isThoughtsOpen ? '▼' : '▶'}
                  </span>
                  <span>
                    {!message.content
                      ? '🧠 Processing...'
                      : `✅ ${message.thoughts.length} steps completed`}
                  </span>
                </button>

                {isThoughtsOpen && (
                  <div className="thoughts-list">
                    {message.thoughts.map((thought, idx) => {
                      const isLast = idx === message.thoughts.length - 1;
                      const isPending = isLast && isStreaming && !message.content;
                      return (
                        <div key={idx} style={styles.thoughtItem}>
                          {isPending ? (
                            <Loader size={14} style={styles.thoughtIconPending} />
                          ) : (
                            <CheckCircle size={14} style={styles.thoughtIconDone} />
                          )}
                          <span>{thought}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {message.content ? (
              <ReactMarkdown
                components={{
                  p: ({ children }) => <p style={styles.messageText}>{renderTextWithCitations({ children })}</p>,
                  strong: ({ children }) => <strong style={styles.bold}>{children}</strong>,
                  em: ({ children }) => <em style={styles.italic}>{children}</em>,
                  ul: ({ children }) => <ul style={styles.list}>{children}</ul>,
                  ol: ({ children }) => <ol style={styles.list}>{children}</ol>,
                  li: ({ children }) => <li style={styles.listItem}>{renderTextWithCitations({ children })}</li>,
                  text: renderTextWithCitations,
                }}
              >
                {message.content}
              </ReactMarkdown>
            ) : isStreaming ? (
              <div style={styles.streamingPlaceholder}>
                <Loader size={16} style={styles.spinner} />
                <span>Generating response...</span>
              </div>
            ) : null}
            
            {isStreaming && message.content && (
              <span style={styles.streamingCursor}>▌</span>
            )}

            {/* Pin / Save Note button */}
            {!isStreaming && message.content && onPinNote && (
              <button
                onClick={handlePin}
                disabled={pinned || pinning}
                style={{
                  ...styles.pinButton,
                  ...(pinned ? styles.pinButtonPinned : {}),
                }}
                title={pinned ? 'Saved to notes' : 'Save to notes'}
              >
                {pinning ? (
                  <Loader size={14} style={styles.spinner} />
                ) : pinned ? (
                  <Check size={14} />
                ) : (
                  <Pin size={14} />
                )}
                <span>{pinned ? 'Saved' : 'Save note'}</span>
              </button>
            )}

            {message.sources && message.sources.length > 0 && (
              <div style={styles.sourcesContainer}>
                <p style={styles.sourcesTitle}>Sources:</p>
                <div style={styles.sourcesList}>
                  {message.sources.map((source, index) => (
                    <button
                      key={`${source.document}-${source.page}-${source.chunk_index || index}`}
                      style={styles.sourceChip}
                      onClick={() => onSourceClick && onSourceClick(source)}
                    >
                      <FileText size={14} />
                      <span>
                        {source.document}, p. {source.page}
                        {source.chunk_index !== undefined && ` §${source.chunk_index + 1}`}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <span style={styles.timestamp}>
          {message.timestamp.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </span>
      </div>

      {isUser && (
        <div style={styles.avatarUser}>
          <User size={20} color="#fff" />
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    marginBottom: '8px',
  },
  avatarBot: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    backgroundColor: '#4a90e2',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  avatarUser: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    backgroundColor: '#666',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  bubble: {
    maxWidth: '70%',
    padding: '12px 16px',
    borderRadius: '12px',
    position: 'relative',
  },
  bubbleUser: {
    backgroundColor: '#4a90e2',
    color: '#fff',
  },
  bubbleAssistant: {
    backgroundColor: '#f5f5f5',
    color: '#333',
  },
  bubbleError: {
    backgroundColor: '#ffebee',
    color: '#c62828',
  },
  bubbleStreaming: {
    borderLeft: '3px solid #4a90e2',
  },
  thoughtsContainer: {
    marginBottom: '10px',
    padding: '10px 12px',
    backgroundColor: '#f0f7ff',
    borderRadius: '8px',
    borderLeft: '3px solid #4a90e2',
  },
  thoughtItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '13px',
    color: '#555',
    marginBottom: '4px',
  },
  thoughtIconDone: {
    color: '#4caf50',
    flexShrink: 0,
  },
  thoughtIconPending: {
    color: '#4a90e2',
    flexShrink: 0,
    animation: 'spin 1s linear infinite',
  },
  streamingPlaceholder: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: '#666',
    fontSize: '14px',
  },
  streamingCursor: {
    display: 'inline-block',
    color: '#4a90e2',
    animation: 'blink 1s ease-in-out infinite',
    marginLeft: '2px',
  },
  spinner: {
    animation: 'spin 1s linear infinite',
  },
  messageText: {
    margin: 0,
    lineHeight: '1.5',
    fontSize: '14px',
  },
  bold: {
    fontWeight: '600',
  },
  italic: {
    fontStyle: 'italic',
  },
  list: {
    marginTop: '8px',
    marginBottom: '8px',
    paddingLeft: '20px',
  },
  listItem: {
    marginBottom: '4px',
    fontSize: '14px',
  },
  sourcesContainer: {
    marginTop: '12px',
    paddingTop: '12px',
    borderTop: '1px solid #e0e0e0',
  },
  sourcesTitle: {
    fontSize: '12px',
    fontWeight: '600',
    color: '#666',
    marginBottom: '8px',
    margin: 0,
  },
  sourcesList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
  },
  sourceChip: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    padding: '6px 10px',
    backgroundColor: '#fff',
    border: '1px solid #4a90e2',
    color: '#4a90e2',
    borderRadius: '16px',
    fontSize: '12px',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  timestamp: {
    display: 'block',
    fontSize: '11px',
    marginTop: '6px',
    opacity: 0.7,
  },
  citationLink: {
    borderRadius: '4px',
    padding: '1px 4px',
    margin: '0 1px',
    fontSize: '13px',
    fontWeight: '500',
    transition: 'all 0.2s',
  },
  citationLinkActive: {
    backgroundColor: '#e3f2fd',
    color: '#1976d2',
    cursor: 'pointer',
    textDecoration: 'underline',
  },
  citationLinkInactive: {
    backgroundColor: '#f5f5f5',
    color: '#999',
    cursor: 'default',
  },
  pinButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    marginTop: '8px',
    padding: '4px 10px',
    fontSize: '12px',
    fontWeight: '500',
    color: '#666',
    backgroundColor: 'transparent',
    border: '1px solid #ddd',
    borderRadius: '16px',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  pinButtonPinned: {
    color: '#2e7d32',
    borderColor: '#4caf50',
    backgroundColor: '#e8f5e9',
    cursor: 'default',
  },
};

export default MessageBubble;
