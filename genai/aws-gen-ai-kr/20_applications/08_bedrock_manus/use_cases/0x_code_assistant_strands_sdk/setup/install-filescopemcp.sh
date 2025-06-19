#!/bin/bash

# FileScopeMCP Installation Script
# For Linux and macOS only

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check and install prerequisites
if ! command -v git &> /dev/null || ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Installing prerequisites...${NC}"
    
    # Check if install-prerequisites.sh exists
    if [ -f "install-prerequisites.sh" ]; then
        chmod +x install-prerequisites.sh
        ./install-prerequisites.sh
    else
        echo -e "${RED}Error:${NC} install-prerequisites.sh not found"
        echo "Please create install-prerequisites.sh first"
        exit 1
    fi
fi

# Clone and build
echo -e "${GREEN}Installing FileScopeMCP...${NC}"

[ -d "FileScopeMCP" ] && rm -rf FileScopeMCP

git clone https://github.com/admica/FileScopeMCP.git
cd FileScopeMCP

chmod +x build.sh
./build.sh

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Copy mcp.json to your project's .cursor directory"
echo "2. Update --base-dir to your project path"
echo "3. Restart Cursor"