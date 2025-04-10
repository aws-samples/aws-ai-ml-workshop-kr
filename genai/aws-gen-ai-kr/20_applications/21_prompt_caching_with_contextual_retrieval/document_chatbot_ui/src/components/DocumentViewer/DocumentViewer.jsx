import React, { useState, useEffect, useRef } from 'react';
import { Box, Typography, Paper, Button, CircularProgress, ButtonGroup } from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import './DocumentViewer.css';

const DocumentViewer = ({
  document,
  onNext,
  onPrevious,
  hasMultipleSources,
  currentIndex,
  totalSources,
  cloudfrontDomain // Accept cloudfrontDomain as a prop instead of using environment variable
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const documentCache = useRef(new Map());
  const iframeRef = useRef(null);

  // Convert S3 URI to CloudFront URL
  const getCloudFrontUrl = (s3Uri) => {
    if (!s3Uri || !cloudfrontDomain) return s3Uri;

    // Check if it's an S3 URI
    if (s3Uri.startsWith('s3://')) {
      console.log("Converting S3 URI to CloudFront URL:", s3Uri);
      // Extract the key (path) from the S3 URI
      // Format: s3://bucket-name/path/to/file.pdf
      const s3Parts = s3Uri.replace('s3://', '').split('/');
      const bucketName = s3Parts[0];
      const keyPath = s3Parts.slice(1).join('/');

      // Return CloudFront URL
      return `https://${cloudfrontDomain}/${keyPath}`;
    }

    // If it's already an HTTPS URL (maybe already CloudFront), return as is
    return s3Uri;
  };

  // Generate a cache key based on URL instead of just ID
  const getCacheKey = (doc) => {
    if (!doc) return null;
    // Use both ID and URL to ensure uniqueness
    return `${doc.id}-${doc.url}`;
  };

  useEffect(() => {
    if (!document) return;

    const cacheKey = getCacheKey(document);
    console.log("Current document cache key:", cacheKey);

    // If we already have this document in cache, don't show loading state
    if (documentCache.current.has(cacheKey)) {
      console.log("Using cached document:", cacheKey);
      setLoading(false);
      return;
    }

    // Otherwise, mark as loading until iframe loads
    console.log("Loading document:", cacheKey);
    setLoading(true);
    setError(null);

    // Add timeout to detect stalled loads
    const timeoutId = setTimeout(() => {
      if (loading) {
        console.log("Document load timed out:", cacheKey);
        setError('Document load timed out. Please try opening it directly.');
        setLoading(false);
      }
    }, 15000); // 15 second timeout

    return () => clearTimeout(timeoutId);
  }, [document, loading]);

  const handleIframeLoad = () => {
    if (document) {
      const cacheKey = getCacheKey(document);
      console.log(`Cached document ${cacheKey}`);
      documentCache.current.set(cacheKey, true);
      setLoading(false);
    }
  };

  const handleIframeError = () => {
    if (document) {
      const cacheKey = getCacheKey(document);
      console.log("Error loading document:", cacheKey);
    }
    setLoading(false);
    setError('Failed to load document. It might be restricted or in an unsupported format.');
  };

  if (!document) {
    return (
      <Box className="document-placeholder">
        <Typography variant="h6">No Source Document</Typography>
        <Typography variant="body1">
          Ask a question in the chat to see relevant source documents here.
        </Typography>
      </Box>
    );
  }

  // Convert S3 URI to CloudFront URL
  const documentUrl = getCloudFrontUrl(document.url);
  const isPdf = documentUrl.toLowerCase().endsWith('.pdf');
  const cacheKey = getCacheKey(document);

  console.log("Document URL after conversion:", documentUrl);

  return (
    <div className="document-viewer">
      <Paper elevation={0} className="document-header">
        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
          {/* Add navigation buttons in header */}
          {hasMultipleSources && (
            <ButtonGroup variant="outlined" size="small" sx={{ mr: 2 }}>
              <Button onClick={onPrevious} size="small">
                <ArrowBackIosNewIcon fontSize="small" />
              </Button>
              <Button disabled size="small">
                {currentIndex} / {totalSources}
              </Button>
              <Button onClick={onNext} size="small">
                <ArrowForwardIosIcon fontSize="small" />
              </Button>
            </ButtonGroup>
          )}
          <Typography variant="h6" sx={{ flex: 1 }}>{document.title}</Typography>
          <Button
            href={documentUrl}
            target="_blank"
            rel="noopener noreferrer"
            endIcon={<OpenInNewIcon />}
            variant="outlined"
            size="small"
          >
            Open
          </Button>
        </Box>
      </Paper>

      <div className="document-content">
        {loading && (
          <Box className="loading-container">
            <CircularProgress />
            <Typography variant="body2">Loading document...</Typography>
            <Button
              href={documentUrl}
              target="_blank"
              variant="contained"
              sx={{ mt: 2 }}
              endIcon={<OpenInNewIcon />}
            >
              Open Document Directly
            </Button>
          </Box>
        )}

        {error && (
          <Box className="error-container">
            <Typography variant="body1" color="error">{error}</Typography>
            <Button
              href={documentUrl}
              target="_blank"
              variant="contained"
              sx={{ mt: 2 }}
              endIcon={<OpenInNewIcon />}
            >
              Open Document Directly
            </Button>
          </Box>
        )}

        {isPdf ? (
          <iframe
            ref={iframeRef}
            src={`${documentUrl}#toolbar=0`}
            title={document.title}
            width="100%"
            height="100%"
            onLoad={handleIframeLoad}
            onError={handleIframeError}
            style={{ display: loading && !documentCache.current.has(cacheKey) ? 'none' : 'block' }}
          />
        ) : (
          <Box className="s3-document-info">
            <Typography variant="body1">
              This document is stored at: <code>{documentUrl}</code>
            </Typography>
            <Typography variant="body2" sx={{ mt: 2 }}>
              Click "Open" to view the original document.
            </Typography>
          </Box>
        )}
      </div>
    </div>
  );
};

export default DocumentViewer;