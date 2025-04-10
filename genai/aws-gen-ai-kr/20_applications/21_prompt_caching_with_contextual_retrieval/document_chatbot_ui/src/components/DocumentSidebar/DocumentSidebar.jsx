import React, { useState, useEffect } from 'react';
import {
  Box,
  Drawer,
  IconButton,
  useMediaQuery,
  useTheme,
  Divider,
  Typography
} from '@mui/material';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';

import DocumentUpload from './DocumentUpload';
import DocumentList from './DocumentList';

const DRAWER_WIDTH = 350;

const Sidebar = ({
  apiUrl,
  cloudFrontDomain,
  onSelectDocument,
  onUploadSuccess,
  open,
  toggleSidebar
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [selectedTab, setSelectedTab] = useState('documents');

  // Close drawer on mobile after a document is selected
  const handleDocumentSelect = (document) => {
    if (onSelectDocument) {
      onSelectDocument(document);
    }

    if (isMobile) {
      toggleSidebar();
    }
  };

  // Handle successful upload
  const handleUploadSuccess = (data) => {
    if (onUploadSuccess) {
      onUploadSuccess(data);
    }

    // Switch to the documents tab after upload
    setSelectedTab('documents');
  };

  return (
    <Drawer
      variant={isMobile ? 'temporary' : 'persistent'}
      anchor="left"
      open={open}
      onClose={toggleSidebar}
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          borderRight: '1px solid rgba(0, 0, 0, 0.12)',
          overflow: 'hidden', // Prevent drawer from causing scrollbars
          position: isMobile ? 'fixed' : 'absolute', // Absolute positioning improves layout consistency
          height: '100%',
        },
        '& + .app-container': {
          marginLeft: 0, // Remove gap
        }
      }}
    >
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: 2,
        borderBottom: '1px solid rgba(0, 0, 0, 0.12)'
      }}>
        <Typography variant="h6" component="div">
          Document Manager
        </Typography>
        <IconButton onClick={toggleSidebar}>
          {theme.direction === 'ltr' ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
      </Box>

      <Box sx={{ p: 2, height: 'calc(100% - 64px)', overflowY: 'auto' }}>
        <DocumentUpload
          apiUrl={apiUrl}
          onUploadSuccess={handleUploadSuccess}
        />

        <Divider sx={{ my: 2 }} />

        <DocumentList
          apiUrl={apiUrl}
          cloudFrontDomain={cloudFrontDomain}
          onSelectDocument={handleDocumentSelect}
        />
      </Box>
    </Drawer>
  );
};

export default Sidebar;