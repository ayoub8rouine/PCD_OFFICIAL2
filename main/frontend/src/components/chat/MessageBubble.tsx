import React from 'react';
import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { Message } from '../../types';
import { cn } from '../../utils/cn';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isBot = message.sender === 'bot';
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "max-w-[80%] mb-4",
        isBot ? "self-start" : "self-end"
      )}
    >
      <div
        className={cn(
          "rounded-2xl shadow-sm px-4 py-3",
          isBot 
            ? "bg-white text-secondary-900 rounded-tl-sm" 
            : "bg-primary text-white rounded-tr-sm"
        )}
      >
        {message.text && <p className="mb-2">{message.text}</p>}
        
        {message.image && (
          <div className="mb-2 overflow-hidden rounded-lg">
            <img 
              src={message.image} 
              alt="Message attachment" 
              className="w-full h-auto max-h-[300px] object-contain" 
            />
          </div>
        )}
        
        {message.audio && (
          <div className="mb-2">
            <audio 
              src={message.audio} 
              controls 
              className="w-full" 
            />
          </div>
        )}
        
        <div className={cn(
          "text-xs mt-1",
          isBot ? "text-secondary-500" : "text-primary-100"
        )}>
          {format(message.timestamp, 'h:mm a')}
        </div>
      </div>
    </motion.div>
  );
};