#!/bin/bash

# Install git and node.js for FileScopeMCP

set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Installing git and node.js...${NC}"

# Update and install
sudo apt-get update -y
sudo apt-get install -y git
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify
echo -e "${GREEN}Installed versions:${NC}"
git --version
node --version
npm --version

echo -e "${GREEN}Prerequisites ready!${NC}"