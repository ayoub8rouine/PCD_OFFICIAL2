import React from 'react';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { Button } from '../ui/Button';
import { useChatStore } from '../../store/chatStore';
import { useAuthStore } from '../../store/authStore';
import { LogOut } from 'lucide-react';

export const ChatInterface: React.FC = () => {
  const { clearMessages } = useChatStore();
  const { logout, user } = useAuthStore();
  
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b border-secondary-200 p-4 flex justify-between items-center">
        <div>
          <h1 className="text-xl font-semibold text-secondary-900">Chat Assistant</h1>
          {user && <p className="text-sm text-secondary-500">Logged in as {user.name}</p>}
        </div>
        <div className="flex gap-2">
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={clearMessages}
          >
            Clear Chat
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={logout}
            rightIcon={<LogOut size={16} />}
          >
            Logout
          </Button>
        </div>
      </div>
      
      {/* Messages */}
      <MessageList />
      
      {/* Input */}
      <ChatInput />
    </div>
  );
};