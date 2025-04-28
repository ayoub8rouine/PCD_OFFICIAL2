import React from 'react';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isBot = message.sender === 'bot';
  
  return (
    <div 
      className={`flex w-full my-2 ${isBot ? 'justify-start' : 'justify-end'}`}
    >
      <div 
        className={`max-w-[80%] md:max-w-[70%] p-3 rounded-lg ${
          isBot 
            ? 'bg-white border border-gray-200 text-gray-800 rounded-tl-none' 
            : 'bg-blue-600 text-white rounded-tr-none'
        }`}
      >
        <div className="flex flex-col">
          <div className="break-words">
            {message.text.split('\n').map((line, index) => (
              <React.Fragment key={index}>
                {line}
                {index < message.text.split('\n').length - 1 && <br />}
              </React.Fragment>
            ))}
          </div>
          <div className={`text-xs mt-1 ${isBot ? 'text-gray-500' : 'text-blue-100'}`}>
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;