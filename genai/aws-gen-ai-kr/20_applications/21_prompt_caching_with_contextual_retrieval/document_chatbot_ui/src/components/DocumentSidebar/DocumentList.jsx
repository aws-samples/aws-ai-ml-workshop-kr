import React, { useState, useEffect, useRef } from 'react';
import {
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  ListItemSecondaryAction,
  Avatar,
  IconButton,
  Typography,
  Paper,
  Chip,
  Box,
  CircularProgress,
  Divider,
  Button,
  Collapse,
  Tooltip,
  Grid,
  LinearProgress,
  Stack
} from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';
import RefreshIcon from '@mui/icons-material/Refresh';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import StorageIcon from '@mui/icons-material/Storage';
import SearchIcon from '@mui/icons-material/Search';

const TERMINAL_STATES = ['COMPLETED', 'ERROR'];
const POLLING_INTERVAL = 10000; // 10 seconds

const DocumentList = ({ apiUrl, cloudFrontDomain, onSelectDocument }) => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [isPolling, setIsPolling] = useState(true);
  const pollingIntervalRef = useRef(null);

  // Function to fetch document list
  const fetchDocuments = async () => {
    if (!apiUrl) return;

    setLoading(documents.length === 0); // Only show loading on initial fetch
    setError(null);

    try {
      console.log(`Fetching documents from ${apiUrl}/documents`);
      const response = await fetch(`${apiUrl}/documents`);
      if (!response.ok) {
        throw new Error(`Failed to fetch documents. Status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Received document data:', data);

      // Map opensearchStatus to indexStatus for compatibility
      const processedDocuments = (data.documents || []).map(doc => {
        // Handle both data formats (opensearchStatus and indexStatus)
        const indexStatus = doc.opensearchStatus
          ? {
              knowledge_base: doc.opensearchStatus.kb_index,
              contextual_retrieval: doc.opensearchStatus.cr_index
            }
          : doc.indexStatus;

        return {
          ...doc,
          indexStatus
        };
      });

      setDocuments(processedDocuments);

      // Check if all documents are in terminal states (COMPLETED or ERROR)
      const allDocumentsProcessed = processedDocuments.every(
        doc => TERMINAL_STATES.includes(doc.status)
      );

      // If all documents are processed, we can stop polling
      if (allDocumentsProcessed && processedDocuments.length > 0) {
        console.log('All documents processed, stopping polling');
        setIsPolling(false);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
      setError(`Failed to load documents: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Set up polling effect
  useEffect(() => {
    // Initial fetch
    fetchDocuments();

    // Set up polling if needed
    if (isPolling) {
      pollingIntervalRef.current = setInterval(fetchDocuments, POLLING_INTERVAL);
    }

    // Cleanup function
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [apiUrl, isPolling]);

  // Effect to update polling state based on document statuses
  useEffect(() => {
    const allDocumentsProcessed = documents.length > 0 &&
      documents.every(doc => TERMINAL_STATES.includes(doc.status));

    if (allDocumentsProcessed && isPolling) {
      setIsPolling(false);
      console.log('All documents processed, polling stopped');
    }
  }, [documents, isPolling]);

  // Toggle expand/collapse for a document
  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  // Convert S3 URL to CloudFront URL
  const getDocumentUrl = (s3Url) => {
    if (!s3Url || !cloudFrontDomain) return '#';

    if (s3Url.startsWith('s3://')) {
      const parts = s3Url.replace('s3://', '').split('/');
      const bucketName = parts[0];
      const key = parts.slice(1).join('/');
      return `https://${cloudFrontDomain}/${key}`;
    }

    return s3Url;
  };

  // Render document status chip
  const renderStatusChip = (status) => {
    let color = 'default';
    let label = status;
    let tooltip = '';

    switch (status) {
      case 'UPLOADED':
        color = 'info';
        tooltip = 'Document has been uploaded and is waiting to be processed';
        break;
      case 'PROCESSING':
        color = 'warning';
        label = 'INGESTING';
        tooltip = 'Document is being ingested';
        break;
      case 'INGESTING':
        color = 'warning';
        tooltip = 'Document is being ingested';
        break;
      case 'COMPLETED':
        color = 'success';
        tooltip = 'Document has been successfully processed and is ready for use';
        break;
      case 'ERROR':
        color = 'error';
        tooltip = 'An error occurred during document processing';
        break;
      default:
        color = 'default';
        tooltip = 'Unknown status';
    }

    return (
      <Tooltip title={tooltip}>
        <Chip
          label={label}
          color={color}
          size="small"
          variant="outlined"
        />
      </Tooltip>
    );
  };

  // Calculate Knowledge Base ingestion progress
  const calculateKnowledgeBaseProgress = (indexStatus) => {
    if (!indexStatus) return 0;

    const statusValues = {
      'PENDING': 0,
      'PROCESSING': 50,
      'INGESTING': 75,
      'COMPLETED': 100,
      'ERROR': 0
    };

    return statusValues[indexStatus.knowledge_base] || 0;
  };

  // Render improved process progress with icons and labels
  const renderProcessProgressBar = (indexStatus) => {
    if (!indexStatus) return null;

    const getProgressInfo = (status) => {
      switch (status) {
        case 'COMPLETED':
          return { value: 100, color: 'success.main', text: 'Complete' };
        case 'PROCESSING':
          return { value: 50, color: 'warning.main', text: 'Ingesting', loading: true };
        case 'INGESTING':
          return { value: 75, color: 'warning.main', text: 'Ingesting', loading: true };
        case 'ERROR':
          return { value: 100, color: 'error.main', text: 'Error' };
        case 'PENDING':
        default:
          return { value: 0, color: 'info.main', text: 'Pending' };
      }
    };

    const kbInfo = getProgressInfo(indexStatus.knowledge_base);
    const crInfo = getProgressInfo(indexStatus.contextual_retrieval);

    return (
      <Box sx={{ width: '100%', mt: 1, mb: 1 }}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'medium' }}>
              Processing Status
            </Typography>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <StorageIcon sx={{ mr: 1, color: 'primary.main' }} fontSize="small" />
              <Typography variant="body2" sx={{ fontWeight: 'medium', flex: 1 }}>
                Bedrock Knowledge Base
              </Typography>
              <Chip
                label={kbInfo.text}
                size="small"
                color={kbInfo.color.split('.')[0]}
                variant="outlined"
                sx={{ minWidth: 90, textAlign: 'center' }}
              />
            </Box>
            {kbInfo.loading ? (
              <LinearProgress
                variant="indeterminate"
                sx={{ height: 6, borderRadius: 1, mb: 2 }}
              />
            ) : (
              <LinearProgress
                variant="determinate"
                value={kbInfo.value}
                sx={{
                  height: 6,
                  borderRadius: 1,
                  mb: 2,
                  backgroundColor: 'rgba(0,0,0,0.05)',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: kbInfo.color
                  }
                }}
              />
            )}
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <SearchIcon sx={{ mr: 1, color: 'secondary.main' }} fontSize="small" />
              <Typography variant="body2" sx={{ fontWeight: 'medium', flex: 1 }}>
                Contextual Retrieval
              </Typography>
              <Chip
                label={crInfo.text}
                size="small"
                color={crInfo.color.split('.')[0]}
                variant="outlined"
                sx={{ minWidth: 90, textAlign: 'center' }}
              />
            </Box>
            {crInfo.loading ? (
              <LinearProgress
                variant="indeterminate"
                color="secondary"
                sx={{ height: 6, borderRadius: 1 }}
              />
            ) : (
              <LinearProgress
                variant="determinate"
                value={crInfo.value}
                sx={{
                  height: 6,
                  borderRadius: 1,
                  backgroundColor: 'rgba(0,0,0,0.05)',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: crInfo.color
                  }
                }}
              />
            )}
          </Grid>
        </Grid>
      </Box>
    );
  };

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown';

    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch (e) {
      return timestamp;
    }
  };

  // Format token usage data safely
  const formatTokenUsage = (tokenUsage) => {
    if (!tokenUsage) return 'No token usage data';

    try {
      // If token usage is already a string, just return it
      if (typeof tokenUsage === 'string') return tokenUsage;

      // If it's an object, format it nicely
      if (typeof tokenUsage === 'object') {
        // Create a more human-readable version of the keys
        const readableLabels = {
          'input_tokens': 'Input Tokens',
          'output_tokens': 'Output Tokens',
          'cache_read_input_tokens': 'Cache Read Tokens',
          'cache_write_input_tokens': 'Cache Write Tokens',
          'inputTokens': 'Input Tokens',
          'outputTokens': 'Output Tokens',
          'cacheReadInputTokens': 'Cache Read Tokens',
          'cacheWriteInputTokens': 'Cache Write Tokens',
        };

        // Check if all token values are zero
        const allZero = Object.values(tokenUsage).every(value => {
          const numValue = typeof value === 'number' ? value :
                        (typeof value === 'string' ? parseFloat(value) : 0);
          return numValue === 0;
        });

        if (allZero) {
          return (
            <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
              Token usage not yet available or processing
            </Typography>
          );
        }

        // Format and display token usage values
        return (
          <Grid container spacing={1}>
            {Object.entries(tokenUsage).map(([key, value]) => {
              // Skip any entries with null, undefined, or N/A values
              if (value === null || value === undefined || value === 'N/A') {
                return null;
              }

              const label = readableLabels[key] || key.replace(/_/g, ' ');

              // Convert value to number if possible for formatting
              let displayValue;
              try {
                // Handle DynamoDB-style values: {"N": "34930"}
                if (typeof value === 'object' && value !== null && 'N' in value) {
                  displayValue = parseInt(value.N, 10).toLocaleString();
                } else {
                  // Normal numbers or strings
                  const numValue = typeof value === 'number' ? value :
                                (typeof value === 'string' ? parseFloat(value) : value);
                  displayValue = !isNaN(numValue) ? numValue.toLocaleString() : value.toString();
                }
              } catch (e) {
                displayValue = String(value);
              }

              return (
                <Grid item xs={6} key={key}>
                  <Typography variant="body2">
                    <strong>{label}:</strong> {displayValue}
                  </Typography>
                </Grid>
              );
            })}
          </Grid>
        );
      }

      // Handle DynamoDB-style values at top level: {"N": "34930"}
      if (typeof tokenUsage === 'object' && tokenUsage !== null) {
        // Check for special N format
        const entries = Object.entries(tokenUsage);
        if (entries.length === 1 && entries[0][0] === 'N') {
          return `${parseInt(entries[0][1], 10).toLocaleString()} tokens`;
        }
      }

      // Fallback for any other type
      return JSON.stringify(tokenUsage);
    } catch (e) {
      console.error('Error formatting token usage:', e);
      return 'Error displaying token usage data';
    }
  };

  // Handle refresh button click
  const handleRefresh = () => {
    fetchDocuments();
  };

  // Handle resume polling button click
  const handleResumePolling = () => {
    setIsPolling(true);
  };

  // Handle document selection
  const handleSelectDocument = (document) => {
    if (onSelectDocument) {
      const documentWithUrl = {
        ...document,
        url: getDocumentUrl(document.s3Url),
        title: document.fileName
      };
      onSelectDocument(documentWithUrl);
    }
  };

  return (
    <Paper elevation={0} sx={{ backgroundColor: 'background.default' }}>
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">Documents</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {!isPolling && documents.length > 0 && (
            <Tooltip title="Resume automatic updates">
              <Button
                size="small"
                onClick={handleResumePolling}
                sx={{ mr: 1 }}
              >
                Auto-update
              </Button>
            </Tooltip>
          )}
          <Tooltip title="Refresh document list">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {loading && documents.length === 0 ? (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <CircularProgress size={30} />
          <Typography variant="body2" sx={{ mt: 2 }}>
            Loading documents...
          </Typography>
        </Box>
      ) : error ? (
        <Box sx={{ p: 2 }}>
          <Typography color="error">{error}</Typography>
        </Box>
      ) : documents.length === 0 ? (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="body1">
            No documents uploaded yet
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Upload a PDF to get started
          </Typography>
        </Box>
      ) : (
        <List sx={{ maxHeight: '500px', overflowY: 'auto' }}>
          {documents.map((doc, index) => (
            <React.Fragment key={doc.id}>
              {index > 0 && <Divider component="li" />}
              <ListItem
                button
                onClick={() => handleSelectDocument(doc)}
                sx={{
                  bgcolor: 'white',
                  ':hover': {
                    bgcolor: 'rgba(0, 0, 0, 0.04)'
                  },
                  // Highlight processing documents
                  ...(doc.status === 'PROCESSING' || doc.status === 'INGESTING' ? {
                    bgcolor: 'rgba(255, 152, 0, 0.05)',
                  } : {})
                }}
              >
                <ListItemAvatar>
                  <Avatar>
                    <DescriptionIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Typography variant="subtitle1" noWrap sx={{ maxWidth: '80%' }}>
                      {doc.fileName}
                    </Typography>
                  }
                  secondary={
                    <React.Fragment>
                      <Typography component="span" variant="body2" color="text.secondary">
                        {formatTimestamp(doc.uploadTime)}
                      </Typography>

                      <Box sx={{ mt: 0.5, display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        {renderStatusChip(doc.status)}
                        {doc.statusMessage && (
                          <Tooltip title={doc.statusMessage}>
                            <IconButton size="small" sx={{ ml: 0.5 }}>
                              <HelpOutlineIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>

                      {/* Knowledge Base Ingestion progress */}
                      {doc.indexStatus && doc.status !== 'COMPLETED' && doc.status !== 'ERROR' && (
                        <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ mr: 1, minWidth: '50px' }}>
                            KB Status:
                          </Typography>
                          <Box sx={{ flex: 1 }}>
                            <LinearProgress
                              variant={['PROCESSING', 'INGESTING'].includes(doc.indexStatus.knowledge_base) ? "indeterminate" : "determinate"}
                              value={calculateKnowledgeBaseProgress(doc.indexStatus)}
                              sx={{ height: 4, borderRadius: 1 }}
                            />
                          </Box>
                        </Box>
                      )}
                    </React.Fragment>
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton edge="end" onClick={(e) => {
                    e.stopPropagation();
                    toggleExpand(doc.id);
                  }}>
                    {expandedId === doc.id ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
              <Collapse in={expandedId === doc.id} timeout="auto" unmountOnExit>
                <Box sx={{ p: 2, pl: 9, bgcolor: 'rgba(0, 0, 0, 0.02)' }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Status: <strong>{doc.status}</strong>
                  </Typography>

                  {doc.statusMessage && (
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {doc.statusMessage}
                    </Typography>
                  )}

                  {/* Knowledge Base Process Status */}
                  {doc.indexStatus && (
                    <Box sx={{ mb: 2, mt: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <StorageIcon sx={{ mr: 1, color: 'primary.main' }} fontSize="small" />
                        <Typography variant="subtitle2" sx={{ fontWeight: 'medium', flex: 1 }}>
                          Bedrock Knowledge Base Status
                        </Typography>
                        <Chip
                          label={doc.indexStatus.knowledge_base}
                          size="small"
                          color={doc.indexStatus.knowledge_base === 'COMPLETED' ? 'success' :
                                 doc.indexStatus.knowledge_base === 'ERROR' ? 'error' : 'warning'}
                          variant="outlined"
                          sx={{ minWidth: 90, textAlign: 'center' }}
                        />
                      </Box>
                      {['PROCESSING', 'INGESTING'].includes(doc.indexStatus.knowledge_base) ? (
                        <LinearProgress
                          variant="indeterminate"
                          sx={{ height: 6, borderRadius: 1 }}
                        />
                      ) : doc.indexStatus.knowledge_base === 'COMPLETED' ? (
                        <LinearProgress
                          variant="determinate"
                          value={100}
                          color="success"
                          sx={{ height: 6, borderRadius: 1 }}
                        />
                      ) : doc.indexStatus.knowledge_base === 'ERROR' ? (
                        <LinearProgress
                          variant="determinate"
                          value={100}
                          color="error"
                          sx={{ height: 6, borderRadius: 1 }}
                        />
                      ) : (
                        <LinearProgress
                          variant="determinate"
                          value={0}
                          sx={{ height: 6, borderRadius: 1 }}
                        />
                      )}
                    </Box>
                  )}

                  {doc.ingestionJobId && (
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Ingestion Job ID:</strong> <code>{doc.ingestionJobId}</code><br />
                      <strong>Ingestion Status:</strong> {doc.ingestionStatus || 'Unknown'}
                    </Typography>
                  )}

                  {/* Contextual Retrieval Status */}
                  {doc.indexStatus && (
                    <Box sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <SearchIcon sx={{ mr: 1, color: 'secondary.main' }} fontSize="small" />
                        <Typography variant="subtitle2" sx={{ fontWeight: 'medium', flex: 1 }}>
                          Contextual Retrieval Status
                        </Typography>
                        <Chip
                          label={doc.indexStatus.contextual_retrieval}
                          size="small"
                          color={doc.indexStatus.contextual_retrieval === 'COMPLETED' ? 'success' :
                                 doc.indexStatus.contextual_retrieval === 'ERROR' ? 'error' : 'warning'}
                          variant="outlined"
                          sx={{ minWidth: 90, textAlign: 'center' }}
                        />
                      </Box>
                      {['PROCESSING', 'INGESTING'].includes(doc.indexStatus.contextual_retrieval) ? (
                        <LinearProgress
                          variant="indeterminate"
                          color="secondary"
                          sx={{ height: 6, borderRadius: 1 }}
                        />
                      ) : doc.indexStatus.contextual_retrieval === 'COMPLETED' ? (
                        <LinearProgress
                          variant="determinate"
                          value={100}
                          color="success"
                          sx={{ height: 6, borderRadius: 1 }}
                        />
                      ) : doc.indexStatus.contextual_retrieval === 'ERROR' ? (
                        <LinearProgress
                          variant="determinate"
                          value={100}
                          color="error"
                          sx={{ height: 6, borderRadius: 1 }}
                        />
                      ) : (
                        <LinearProgress
                          variant="determinate"
                          value={0}
                          sx={{ height: 6, borderRadius: 1 }}
                        />
                      )}
                    </Box>
                  )}

                  {/* Token Usage */}
                  {doc.tokenUsage && (
                    <Box sx={{ mt: 2, mb: 1, p: 1.5, bgcolor: 'rgba(0, 0, 0, 0.03)', borderRadius: 1, border: '1px solid rgba(0, 0, 0, 0.08)' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <SearchIcon sx={{ mr: 1, color: 'secondary.main' }} fontSize="small" />
                        <Typography variant="subtitle2">
                          Contextual Retrieval Token Usage
                        </Typography>
                      </Box>
                      <Box sx={{ ml: 1, mt: 0.5, fontSize: '0.85rem' }}>
                        {formatTokenUsage(doc.tokenUsage)}
                      </Box>
                    </Box>
                  )}

                  <Button
                    variant="outlined"
                    size="small"
                    endIcon={<OpenInNewIcon />}
                    href={getDocumentUrl(doc.s3Url)}
                    target="_blank"
                    onClick={(e) => e.stopPropagation()}
                    sx={{ mt: 1 }}
                  >
                    View Document
                  </Button>
                </Box>
              </Collapse>
            </React.Fragment>
          ))}
        </List>
      )}
    </Paper>
  );
};

export default DocumentList;