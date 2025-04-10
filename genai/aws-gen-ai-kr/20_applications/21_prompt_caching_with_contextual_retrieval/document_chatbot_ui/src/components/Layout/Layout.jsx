import React, { useState, useEffect } from 'react';
import Chat from '../Chat/Chat';
import DocumentViewer from '../DocumentViewer/DocumentViewer';
import Selector from '../Selector/Selector';
import Sidebar from '../DocumentSidebar/DocumentSidebar';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import {
  CircularProgress,
  Box,
  Typography,
  Button,
  ButtonGroup,
  Paper,
  IconButton,
  AppBar,
  Toolbar,
  useMediaQuery
} from '@mui/material';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import MenuIcon from '@mui/icons-material/Menu';
import './Layout.css';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2563eb',
    },
    secondary: {
      main: '#3b82f6',
    },
    background: {
      default: '#f8fafc',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

// Sidebar drawer width
const DRAWER_WIDTH = 350;

function Layout({ config }) {
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [messages, setMessages] = useState([]);
  const [availableSources, setAvailableSources] = useState([]);
  const [currentDocumentIndex, setCurrentDocumentIndex] = useState(0);
  const [wsConnection, setWsConnection] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(true);
  const [connectionError, setConnectionError] = useState(null);
  const [aiRespondingMessageId, setAiRespondingMessageId] = useState(null);
  const [selectedModel, setSelectedModel] = useState('amazon.nova-pro-v1:0');
  const [selectedSearchMethod, setSelectedSearchMethod] = useState('opensearch');
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);

  // Current document derived from available sources and current index
  const currentDocument = availableSources.length > 0 ? availableSources[currentDocumentIndex] : null;

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // Function to handle model changes
  const handleModelChange = (modelId) => {
    setSelectedModel(modelId);
  };

  // Function to handle search method changes
  const handleSearchMethodChange = (methodId) => {
    setSelectedSearchMethod(methodId);
  };

  // Function to navigate between documents
  const navigateDocument = (direction) => {
    if (availableSources.length === 0) return;

    setCurrentDocumentIndex(prevIndex => {
      if (direction === 'next') {
        return (prevIndex + 1) % availableSources.length;
      } else {
        return prevIndex === 0 ? availableSources.length - 1 : prevIndex - 1;
      }
    });
  };

  // Handle document selection from sidebar
  const handleSelectDocument = (document) => {
    // Check if this document is already in available sources
    const existingIndex = availableSources.findIndex(
      source => source.url === document.url
    );

    if (existingIndex >= 0) {
      // If exists, just focus on it
      setCurrentDocumentIndex(existingIndex);
    } else {
      // Otherwise add it and focus on it
      setAvailableSources(prev => [...prev, document]);
      setCurrentDocumentIndex(availableSources.length);
    }
  };

  // Handle successful document upload
  const handleUploadSuccess = (data) => {
    // We'll potentially refetch documents, but the regular polling will handle that
    console.log('Document uploaded successfully:', data);
  };

  // Connect to WebSocket
  useEffect(() => {
    if (!config || !config.websocketUrl) return;

    const wsUrl = config.websocketUrl;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Connected to WebSocket');
      setIsConnected(true);
      setIsConnecting(false);
      setWsConnection(ws);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("WebSocket message received:", data);

      // Extract the text content from wherever it might be in the response
      let textContent = null;
      if (data.type === 'text' && data.content) {
        textContent = data.content;
      } else if (data.text) {
        textContent = data.text;
      } else if (data.output && data.output.text) {
        textContent = data.output.text;
      }

      // If we have text content to display
      if (textContent !== null) {
        setMessages(prevMessages => {
          // If we already have an AI message we're building
          if (aiRespondingMessageId !== null) {
            // Return a new array with the AI message text updated
            return prevMessages.map(msg =>
              msg.id === aiRespondingMessageId
                ? {...msg, text: msg.text + textContent}
                : msg
            );
          } else {
            // Create a new AI message and store its ID
            const newMessageId = Date.now();
            setAiRespondingMessageId(newMessageId);

            return [...prevMessages, {
              id: newMessageId,
              text: textContent,
              sender: 'ai',
              timestamp: new Date().toISOString()
            }];
          }
        });
      }

      // Handle citation (newer format)
      if (data.type === 'citation' && data.sourceUrl) {
        // Check if this source is already in our list
        const existingIndex = availableSources.findIndex(source => source.id === data.sourceId);

        if (existingIndex >= 0) {
          // Source already exists, make it the current one
          setCurrentDocumentIndex(existingIndex);
        } else {
          // Add new source to the list
          const newSource = {
            id: data.sourceId,
            url: data.sourceUrl,
            title: `Source ${data.sourceId}`
          };

          // Use functional update to correctly set the index
          setAvailableSources(prev => {
            const newSources = [...prev, newSource];
            setCurrentDocumentIndex(newSources.length - 1);
            return newSources;
          });
        }
      }

      // Handle 'complete' message - signals end of AI response
      if (data.type === 'complete') {
        // Reset the AI responding message ID for the next interaction
        setAiRespondingMessageId(null);

        // Only update sources if we haven't received any yet
        if (data.sources && Object.keys(data.sources).length > 0 && availableSources.length === 0) {
          const completeSources = Object.entries(data.sources).map(([id, url]) => ({
            id,
            url,
            title: `Source ${id}`
          }));

          setAvailableSources(completeSources);
          setCurrentDocumentIndex(0);
        }
      }

      // Handle errors
      if (data.error || (data.type === 'error' && data.message)) {
        const errorMsg = data.error || data.message;
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: `Error: ${errorMsg}`,
          sender: 'system',
          timestamp: new Date().toISOString()
        }]);
        setAiRespondingMessageId(null);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from WebSocket');
      setIsConnected(false);
      setIsConnecting(false);
      setAiRespondingMessageId(null);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionError('Failed to connect to the server. Please try again later.');
      setIsConnecting(false);
      setAiRespondingMessageId(null);
    };

    return () => {
      if (ws) ws.close();
    };
  }, [config]);

  const handleSendMessage = (message) => {
    if (wsConnection && isConnected) {
      // Reset the AI responding message ID for a new conversation turn
      setAiRespondingMessageId(null);

      // Add user message to chat
      setMessages(prev => [...prev, {
        id: Date.now(),
        text: message,
        sender: 'user',
        timestamp: new Date().toISOString()
      }]);

      // Send to WebSocket with model and search method information
      wsConnection.send(JSON.stringify({
        query: message,
        modelArn: selectedModel,
        searchMethod: selectedSearchMethod
      }));
    }
  };

  // If loading or connecting, show a loading indicator
  if (isConnecting && !connectionError) {
    return (
      <ThemeProvider theme={theme}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
          <CircularProgress />
        </Box>
      </ThemeProvider>
    );
  }

  // If connection error, show error message
  if (connectionError) {
    return (
      <ThemeProvider theme={theme}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
          <Typography variant="h6" color="error">
            {connectionError}
          </Typography>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <div className="app-layout">
        {/* Top AppBar */}
        <AppBar position="static" color="default" elevation={1}>
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={toggleSidebar}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>

            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Document Chatbot
            </Typography>

            {/* Move selector to the AppBar for cleaner layout */}
            <Box sx={{ display: { xs: 'none', md: 'block' } }}>
              <Selector
                selectedModel={selectedModel}
                onModelChange={handleModelChange}
                selectedMethod={selectedSearchMethod}
                onMethodChange={handleSearchMethodChange}
              />
            </Box>
          </Toolbar>
        </AppBar>

        {/* Responsive Selector for mobile */}
        <Box sx={{ display: { xs: 'block', md: 'none' }, p: 2 }}>
          <Selector
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
            selectedMethod={selectedSearchMethod}
            onMethodChange={handleSearchMethodChange}
          />
        </Box>

        {/* Main content */}
        <Box className="app-container" sx={{
          ml: sidebarOpen && !isMobile ? `${DRAWER_WIDTH}px` : 0,
          transition: theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          overflow: 'hidden', // Prevent scrolling of the container
          gap: 0, // Remove any gap between the sidebar and chat panel
        }}>
          {/* Sidebar */}
          <Sidebar
            apiUrl={config?.uploadApiUrl}
            cloudFrontDomain={config?.cloudfrontDomain}
            onSelectDocument={handleSelectDocument}
            onUploadSuccess={handleUploadSuccess}
            open={sidebarOpen}
            toggleSidebar={toggleSidebar}
          />

          {/* Chat panel */}
          <div className="chat-panel">
            <Chat
              messages={messages}
              onSendMessage={handleSendMessage}
              isConnected={isConnected}
            />
          </div>

          {/* Document panel */}
          <div className="document-panel">
            <DocumentViewer
              document={currentDocument}
              onNext={() => navigateDocument('next')}
              onPrevious={() => navigateDocument('prev')}
              hasMultipleSources={availableSources.length > 1}
              currentIndex={currentDocumentIndex + 1}
              totalSources={availableSources.length}
              cloudfrontDomain={config?.cloudfrontDomain}
            />

            {/* Navigation footer */}
            <div className="document-navigation">
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 1 }}>
                <ButtonGroup variant="outlined" size="small">
                  <Button
                    onClick={() => navigateDocument('prev')}
                    startIcon={<ArrowBackIosNewIcon />}
                    disabled={availableSources.length <= 1}
                  >
                    Prev
                  </Button>
                  <Button disabled>
                    {availableSources.length > 0 ?
                      `${currentDocumentIndex + 1} / ${availableSources.length}` :
                      "0 / 0"}
                  </Button>
                  <Button
                    onClick={() => navigateDocument('next')}
                    endIcon={<ArrowForwardIosIcon />}
                    disabled={availableSources.length <= 1}
                  >
                    Next
                  </Button>
                </ButtonGroup>
              </Box>
            </div>
          </div>
        </Box>
      </div>
    </ThemeProvider>
  );
}

export default Layout;