import React, { useState, useEffect } from 'react';
import Layout from './components/Layout/Layout';
import './components/Layout/Layout.css';
import { CircularProgress, Box, Typography } from '@mui/material';

function App() {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch('/config.json');
        if (!response.ok) {
          throw new Error(`Failed to load configuration: ${response.statusText}`);
        }

        const configData = await response.json();

        setConfig(configData);
        setLoading(false);
      } catch (error) {
        console.error('Error loading configuration:', error);
        setError('Failed to load application configuration. Please refresh the page or contact support.');
        setLoading(false);
      }
    };

    loadConfig();
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
        <Typography variant="h6" color="error">
          {error}
        </Typography>
      </Box>
    );
  }

  return <Layout config={config} />;
}

export default App;