import React, { useState } from 'react';
import {
  Button,
  Box,
  Typography,
  CircularProgress,
  Paper,
  Alert
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const DocumentUpload = ({ apiUrl, onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      // Check if the file is a PDF
      if (selectedFile.type !== 'application/pdf') {
        setUploadError('Only PDF files are supported');
        setFile(null);
        return;
      }

      // Check file size (limit to 10MB)
      if (selectedFile.size > 10 * 1024 * 1024) {
        setUploadError('File size cannot exceed 10MB');
        setFile(null);
        return;
      }

      setFile(selectedFile);
      setUploadError(null);
      setUploadSuccess(false);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadError('Please select a file to upload');
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(false);

    try {
      // Convert file to base64
      const reader = new FileReader();

      reader.onload = async (e) => {
        const base64File = e.target.result;

        // Prepare the payload
        const payload = {
          fileName: file.name,
          fileType: file.type,
          file: base64File
        };

        // Send the file to the API
        const response = await fetch(`${apiUrl}/uploads`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Upload failed');
        }

        const data = await response.json();
        setUploadSuccess(true);

        // Call callback function if provided
        if (onUploadSuccess) {
          onUploadSuccess(data);
        }
      };

      reader.onerror = () => {
        throw new Error('Failed to read file');
      };

      reader.readAsDataURL(file);
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadError(error.message || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Paper elevation={0} sx={{ p: 2, mb: 3, backgroundColor: 'background.default' }}>
      <Typography variant="h6" gutterBottom>
        Upload Document
      </Typography>

      <Box
        sx={{
          border: '2px dashed #ccc',
          borderRadius: 2,
          p: 3,
          textAlign: 'center',
          mb: 2,
          backgroundColor: 'white'
        }}
      >
        <input
          accept="application/pdf"
          style={{ display: 'none' }}
          id="upload-file"
          type="file"
          onChange={handleFileChange}
          disabled={uploading}
        />
        <label htmlFor="upload-file">
          <Button
            variant="outlined"
            component="span"
            startIcon={<CloudUploadIcon />}
            disabled={uploading}
            sx={{ mb: 2 }}
          >
            Select PDF
          </Button>
        </label>

        {file && (
          <Typography variant="body2" sx={{ mt: 1 }}>
            Selected: {file.name} ({Math.round(file.size / 1024)} KB)
          </Typography>
        )}
      </Box>

      {uploadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {uploadError}
        </Alert>
      )}

      {uploadSuccess && (
        <Alert severity="success" sx={{ mb: 2 }}>
          File uploaded successfully!
        </Alert>
      )}

      <Button
        variant="contained"
        color="primary"
        onClick={handleUpload}
        disabled={!file || uploading}
        fullWidth
      >
        {uploading ? (
          <>
            <CircularProgress size={24} color="inherit" sx={{ mr: 1 }} />
            Uploading...
          </>
        ) : (
          'Upload Document'
        )}
      </Button>
    </Paper>
  );
};

export default DocumentUpload;