import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
  ConversationHeader,
} from '@chatscope/chat-ui-kit-react';
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import { Box, Typography, Avatar, Badge } from '@mui/material';
import SmartToyOutlinedIcon from '@mui/icons-material/SmartToyOutlined';
import PersonOutlineIcon from '@mui/icons-material/PersonOutline';
import './Chat.css';

const Chat = ({ messages, onSendMessage, isConnected }) => {
  const [inputValue, setInputValue] = useState('');
  const messageListRef = useRef(null);

  // Auto-scroll to the bottom when new messages arrive
  useEffect(() => {
    if (messageListRef.current) {
      const element = messageListRef.current;
      element.scrollTop = element.scrollHeight;
    }
  }, [messages]);

  const handleSend = (message) => {
    onSendMessage(message);
    setInputValue('');
  };

  // Group messages by sender and combine those that are close in time
  const groupedMessages = useMemo(() => {
    if (!messages.length) return [];

    const result = [];
    let currentGroup = null;

    messages.forEach(msg => {
      // Start a new group if:
      // 1. This is the first message
      // 2. The sender changed
      // 3. More than 2 seconds passed between messages
      const shouldStartNewGroup =
        !currentGroup ||
        currentGroup.sender !== msg.sender ||
        (new Date(msg.timestamp).getTime() - new Date(currentGroup.timestamp).getTime() > 2000);

      if (shouldStartNewGroup) {
        // Add previous group to result if it exists
        if (currentGroup) {
          result.push(currentGroup);
        }

        // Start new group
        currentGroup = {
          id: msg.id,
          text: msg.text,
          sender: msg.sender,
          timestamp: msg.timestamp
        };
      } else {
        // Append to existing group
        currentGroup.text += msg.text;
        currentGroup.timestamp = msg.timestamp; // Update timestamp to most recent
      }
    });

    // Add the last group
    if (currentGroup) {
      result.push(currentGroup);
    }

    return result;
  }, [messages]);

  return (
    <div className="chat-container">
      <MainContainer>
        <ChatContainer>
          <ConversationHeader>
            <ConversationHeader.Content>
              <Typography variant="h6">Talk to your documents</Typography>
            </ConversationHeader.Content>
            <ConversationHeader.Actions>
              {isConnected ? (
                <span className="status-indicator connected">Connected</span>
              ) : (
                <span className="status-indicator disconnected">Disconnected</span>
              )}
            </ConversationHeader.Actions>
          </ConversationHeader>

          <MessageList ref={messageListRef}>
            {messages.length === 0 && (
              <Box className="empty-state">
                <Typography variant="h6">Welcome to Document Chat</Typography>
                <Typography variant="body1">
                  Ask questions about your documents and I'll find the answers for you.
                </Typography>
              </Box>
            )}

            {groupedMessages.map((msg) => (
              <Message
                key={msg.id}
                model={{
                  message: msg.text,
                  sentTime: new Date(msg.timestamp).toLocaleTimeString(),
                  sender: msg.sender === 'user' ? 'You' : 'AI',
                  direction: msg.sender === 'user' ? 'outgoing' : 'incoming',
                  position: 'single'
                }}
              >
                {msg.sender === 'user' ? (
                  <Avatar
                    sx={{
                      bgcolor: '#3b82f6',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                    }}
                    className="user-avatar-icon"
                  >
                    <PersonOutlineIcon />
                  </Avatar>
                ) : (
                  <Badge
                    overlap="circular"
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                    variant="dot"
                    sx={{ '& .MuiBadge-badge': { backgroundColor: '#44b700' } }}
                  >
                    <Avatar
                      sx={{
                        bgcolor: '#7c3aed',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                      }}
                      className="ai-avatar-icon"
                    >
                      <SmartToyOutlinedIcon />
                    </Avatar>
                  </Badge>
                )}
              </Message>
            ))}
          </MessageList>

          <MessageInput
            placeholder={isConnected ? "Type your question here..." : "Connection lost..."}
            onSend={handleSend}
            value={inputValue}
            onChange={setInputValue}
            disabled={!isConnected}
            attachButton={false}
          />
        </ChatContainer>
      </MainContainer>
    </div>
  );
};

export default Chat;
