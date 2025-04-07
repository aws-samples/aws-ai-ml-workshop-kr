import React from 'react';
import { FormControl, InputLabel, Select, MenuItem, Box, Grid } from '@mui/material';

const Selector = ({ selectedModel, onModelChange, selectedMethod, onMethodChange }) => {
  const models = [
    { id: 'amazon.nova-pro-v1:0', name: 'Amazon Nova Pro' },
    { id: 'amazon.nova-lite-v1:0', name: 'Amazon Nova Lite' },
    { id: 'anthropic.claude-3-7-sonnet-20250219-v1:0', name: 'Claude 3.7 Sonnet' },
    { id: 'anthropic.claude-3-5-sonnet-20241022-v2:0', name: 'Claude 3.5 Sonnet v2' },
    { id: 'anthropic.claude-3-5-haiku-20241022-v1:0', name: 'Claude 3.5 Haiku' }
  ];

  const methods = [
    { id: 'opensearch', name: 'Knowledge Base (OpenSearch)' },
    { id: 'contextual', name: 'Contextual Retrieval' }
  ];

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Box sx={{ minWidth: 220 }}>
          <FormControl fullWidth size="small">
            <InputLabel id="model-select-label">AI Model</InputLabel>
            <Select
              labelId="model-select-label"
              id="model-select"
              value={selectedModel}
              label="AI Model"
              onChange={(e) => onModelChange(e.target.value)}
            >
              {models.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  {model.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      </Grid>
      <Grid item xs={12} md={6}>
        <Box sx={{ minWidth: 220 }}>
          <FormControl fullWidth size="small">
            <InputLabel id="search-method-select-label">Search Method</InputLabel>
            <Select
              labelId="search-method-select-label"
              id="search-method-select"
              value={selectedMethod}
              label="Search Method"
              onChange={(e) => onMethodChange(e.target.value)}
            >
              {methods.map((method) => (
                <MenuItem key={method.id} value={method.id}>
                  {method.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      </Grid>
    </Grid>
  );
};

export default Selector;
