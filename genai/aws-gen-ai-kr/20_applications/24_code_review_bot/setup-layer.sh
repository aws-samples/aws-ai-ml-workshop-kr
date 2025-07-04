#!/bin/bash
# setup-layers.sh

# Requests Layer
echo "Setting up requests layer..."
mkdir -p layer/requests/python
cd layer/requests/python
pip3.12 install requests -t .
cd ../../..

# Networkx Layer
echo "Setting up networkx layer..."
mkdir -p layer/networkx/python
cd layer/networkx/python
pip3.12 install networkx numpy -t .
cd ../../..

# Clean up unnecessary files
find layer -type d -name "__pycache__" -exec rm -rf {} +
find layer -type d -name "*.dist-info" -exec rm -rf {} +
find layer -type d -name "*.egg-info" -exec rm -rf {} +
find layer -type f -name "*.pyc" -delete

echo "Layer setup complete!"
