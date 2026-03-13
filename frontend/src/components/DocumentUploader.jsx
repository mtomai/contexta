import React, { useState, useRef } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { uploadDocument } from '../services/api';

const DocumentUploader = ({ onUploadSuccess, notebookId }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file) => {
    // Validate file type
    const allowedTypes = ['.pdf', '.doc', '.docx', '.xlsx'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedTypes.includes(fileExt)) {
      setError('File type not supported. Please select a PDF, Word or Excel file.');
      return;
    }

    // Validate file size (50MB max)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File too large. Maximum size: 50MB');
      return;
    }

    setError(null);
    setSuccess(false);
    setUploading(true);
    setUploadProgress(0);

    try {
      const result = await uploadDocument(file, notebookId, (progress) => {
        setUploadProgress(progress);
      });

      setSuccess(true);
      setUploading(false);
      setUploadProgress(100);

      // Reset after 2 seconds
      setTimeout(() => {
        setSuccess(false);
        setUploadProgress(0);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }, 2000);

      // Notify parent component
      if (onUploadSuccess) {
        onUploadSuccess(result);
      }

    } catch (err) {
      setUploading(false);
      setUploadProgress(0);
      setError(err.response?.data?.detail || 'Error during file upload');
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Upload Documents</h2>

      <div
        style={{
          ...styles.dropZone,
          ...(isDragging ? styles.dropZoneActive : {}),
          ...(uploading ? styles.dropZoneUploading : {})
        }}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.doc,.docx,.xlsx"
          onChange={handleFileSelect}
          style={styles.fileInput}
        />

        {!uploading && !success && (
          <>
            <Upload size={48} color="#666" style={styles.icon} />
            <p style={styles.dropText}>
              Drag a file here or click to select
            </p>
            <p style={styles.dropSubtext}>PDF or Word (max 50MB)</p>
          </>
        )}

        {uploading && (
          <>
            <FileText size={48} color="#4a90e2" style={styles.icon} />
            <p style={styles.dropText}>Uploading...</p>
            <div style={styles.progressBar}>
              <div style={{ ...styles.progressFill, width: `${uploadProgress}%` }} />
            </div>
            <p style={styles.progressText}>{uploadProgress}%</p>
          </>
        )}

        {success && (
          <>
            <CheckCircle size={48} color="#4caf50" style={styles.icon} />
            <p style={styles.successText}>Document uploaded successfully!</p>
          </>
        )}
      </div>

      {error && (
        <div style={styles.errorContainer}>
          <AlertCircle size={18} color="#f44336" />
          <span style={styles.errorText}>{error}</span>
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    padding: '20px',
    backgroundColor: '#fff',
    borderRadius: '8px',
    marginBottom: '20px',
    flexShrink: 0,
  },
  title: {
    fontSize: '18px',
    fontWeight: '600',
    marginBottom: '16px',
    color: '#333',
  },
  dropZone: {
    border: '2px dashed #ccc',
    borderRadius: '8px',
    padding: '40px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.3s',
    backgroundColor: '#fafafa',
  },
  dropZoneActive: {
    borderColor: '#4a90e2',
    backgroundColor: '#e3f2fd',
  },
  dropZoneUploading: {
    cursor: 'default',
  },
  fileInput: {
    display: 'none',
  },
  icon: {
    marginBottom: '12px',
  },
  dropText: {
    fontSize: '16px',
    color: '#666',
    marginBottom: '8px',
  },
  dropSubtext: {
    fontSize: '14px',
    color: '#999',
  },
  progressBar: {
    width: '100%',
    height: '8px',
    backgroundColor: '#e0e0e0',
    borderRadius: '4px',
    overflow: 'hidden',
    marginTop: '12px',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#4a90e2',
    transition: 'width 0.3s',
  },
  progressText: {
    marginTop: '8px',
    fontSize: '14px',
    color: '#666',
  },
  successText: {
    fontSize: '16px',
    color: '#4caf50',
    fontWeight: '500',
  },
  errorContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '12px',
    padding: '12px',
    backgroundColor: '#ffebee',
    borderRadius: '4px',
  },
  errorText: {
    fontSize: '14px',
    color: '#c62828',
  },
};

export default DocumentUploader;
