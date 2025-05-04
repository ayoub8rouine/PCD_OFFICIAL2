import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { MessageBubble } from './MessageBubble';
import { useChatStore } from '../../store/chatStore';

export const MessageList: React.FC = () => {
  const { messages, isLoading } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-secondary-500">
        <p className="text-center">No messages yet. Start a conversation!</p>
      </div>
    );
  }
  
  return (
    <div className="flex-1 overflow-y-auto p-4 flex flex-col">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      
      {isLoading && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="self-start max-w-[80%] mb-4"
        >
          <div className="bg-white rounded-2xl shadow-sm px-4 py-3 rounded-tl-sm">
            <div className="flex space-x-2">
              <div className="h-2 w-2 bg-secondary-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="h-2 w-2 bg-secondary-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              <div className="h-2 w-2 bg-secondary-300 rounded-full animate-bounce" style={{ animationDelay: '600ms' }} />
            </div>
          </div>
        </motion.div>
      )}
      
      <div ref={messagesEndRef} />
    </div>
  );
};