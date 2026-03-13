import React from 'react';
import { X, FileText, BookOpen } from 'lucide-react';

const SourceViewer = ({ source, onClose }) => {
  if (!source) return null;

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div style={styles.backdrop} onClick={handleBackdropClick}>
      <div style={styles.modal}>
        <div style={styles.header}>
          <div style={styles.headerContent}>
            <FileText size={24} color="#4a90e2" />
            <h2 style={styles.title}>Document Source</h2>
          </div>
          <button style={styles.closeButton} onClick={onClose}>
            <X size={24} />
          </button>
        </div>

        <div style={styles.content}>
          <div style={styles.metaSection}>
            <div style={styles.metaItem}>
              <BookOpen size={18} color="#666" />
              <div>
                <p style={styles.metaLabel}>Document</p>
                <p style={styles.metaValue}>{source.document}</p>
              </div>
            </div>

            <div style={styles.metaItem}>
              <FileText size={18} color="#666" />
              <div>
                <p style={styles.metaLabel}>Page</p>
                <p style={styles.metaValue}>{source.page}</p>
              </div>
            </div>

            {source.chunk_index !== undefined && (
              <div style={styles.metaItem}>
                <BookOpen size={18} color="#666" />
                <div>
                  <p style={styles.metaLabel}>Paragraph</p>
                  <p style={styles.metaValue}>#{source.chunk_index + 1}</p>
                </div>
              </div>
            )}

            {source.relevance_score !== undefined && (
              <div style={styles.metaItem}>
                <div style={{
                  ...styles.relevanceIndicator,
                  backgroundColor: getRelevanceColor(source.relevance_score)
                }}>
                  {Math.round(source.relevance_score * 100)}%
                </div>
                <div>
                  <p style={styles.metaLabel}>Relevance</p>
                  <p style={styles.metaValue}>
                    {getRelevanceLabel(source.relevance_score)}
                  </p>
                </div>
              </div>
            )}
          </div>

          <div style={styles.excerptSection}>
            <h3 style={styles.excerptTitle}>Relevant Content:</h3>
            <div style={styles.excerptBox}>
              <p style={styles.excerptText}>{source.chunk_text}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const getRelevanceColor = (score) => {
  // Adjusted thresholds for RAG scenarios where 0.4-0.7 is typical
  if (score >= 0.6) return '#4caf50';  // Good relevance
  if (score >= 0.4) return '#ff9800';  // Medium relevance
  return '#f44336';  // Low relevance
};

const getRelevanceLabel = (score) => {
  if (score >= 0.6) return 'High';
  if (score >= 0.4) return 'Medium';
  return 'Low';
};

const styles = {
  backdrop: {
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
    padding: '20px',
  },
  modal: {
    backgroundColor: '#fff',
    borderRadius: '12px',
    maxWidth: '700px',
    width: '100%',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '20px 24px',
    borderBottom: '1px solid #e0e0e0',
  },
  headerContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  title: {
    fontSize: '20px',
    fontWeight: '600',
    color: '#333',
    margin: 0,
  },
  closeButton: {
    padding: '8px',
    backgroundColor: 'transparent',
    border: 'none',
    cursor: 'pointer',
    borderRadius: '4px',
    transition: 'background-color 0.2s',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    flex: 1,
    overflowY: 'auto',
    padding: '24px',
  },
  metaSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    marginBottom: '24px',
    padding: '16px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
  },
  metaItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  metaLabel: {
    fontSize: '12px',
    color: '#666',
    margin: 0,
    marginBottom: '4px',
  },
  metaValue: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#333',
    margin: 0,
  },
  relevanceIndicator: {
    width: '44px',
    height: '44px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#fff',
    fontSize: '12px',
    fontWeight: '600',
  },
  excerptSection: {
    marginTop: '24px',
  },
  excerptTitle: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#333',
    marginBottom: '12px',
  },
  excerptBox: {
    padding: '16px',
    backgroundColor: '#fafafa',
    border: '1px solid #e0e0e0',
    borderRadius: '8px',
    borderLeft: '4px solid #4a90e2',
  },
  excerptText: {
    fontSize: '14px',
    lineHeight: '1.6',
    color: '#333',
    margin: 0,
    whiteSpace: 'pre-wrap',
  },
};

export default SourceViewer;
