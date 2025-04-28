import React, { useState } from 'react';
import { format } from 'date-fns';
import { MessageSquare, Image as ImageIcon } from 'lucide-react';
import { useChatStore, ChatMessage } from '../store/chatStore';

interface DoctorDashboardProps {
  clients: { id: string; email: string }[];
}

const DoctorDashboard: React.FC<DoctorDashboardProps> = ({ clients }) => {
  const [selectedClient, setSelectedClient] = useState<string | null>(null);
  const { getConversation } = useChatStore();

  const handleClientSelect = (clientId: string) => {
    setSelectedClient(clientId);
  };

  const renderConversation = (messages: ChatMessage[]) => {
    return messages.map((message) => (
      <div
        key={message.id}
        className={`flex ${
          message.sender === 'user' ? 'justify-end' : 'justify-start'
        } mb-4`}
      >
        <div
          className={`max-w-[70%] p-3 rounded-lg ${
            message.sender === 'user'
              ? 'bg-blue-600 text-white'
              : 'bg-white border border-gray-200'
          }`}
        >
          {message.imageUrl && (
            <img
              src={message.imageUrl}
              alt="Uploaded content"
              className="max-w-full rounded-lg mb-2"
            />
          )}
          <p className="break-words">{message.text}</p>
          <p className="text-xs mt-1 opacity-75">
            {format(message.timestamp, 'HH:mm - MMM d, yyyy')}
          </p>
        </div>
      </div>
    ));
  };

  return (
    <div className="flex h-full">
      {/* Clients List */}
      <div className="w-1/4 border-r border-gray-200 bg-white overflow-y-auto">
        <div className="p-4">
          <h2 className="text-lg font-semibold mb-4">Recent Conversations</h2>
          <div className="space-y-2">
            {clients.map((client) => {
              const conversation = getConversation(client.id);
              const lastMessage = conversation[conversation.length - 1];
              const hasImages = conversation.some((msg) => msg.imageUrl);

              return (
                <button
                  key={client.id}
                  onClick={() => handleClientSelect(client.id)}
                  className={`w-full p-3 rounded-lg text-left transition-colors ${
                    selectedClient === client.id
                      ? 'bg-blue-50 border-blue-200'
                      : 'hover:bg-gray-50 border-transparent'
                  } border`}
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-medium">{client.email}</span>
                    {lastMessage && (
                      <span className="text-xs text-gray-500">
                        {format(lastMessage.timestamp, 'HH:mm')}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-500">
                    <MessageSquare size={14} />
                    <span>{conversation.length} messages</span>
                    {hasImages && <ImageIcon size={14} />}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Conversation View */}
      <div className="flex-1 bg-gray-50 p-4 overflow-y-auto">
        {selectedClient ? (
          <div className="max-w-3xl mx-auto">
            {renderConversation(getConversation(selectedClient))}
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-500">
            Select a client to view conversation
          </div>
        )}
      </div>
    </div>
  );
};

export default DoctorDashboard;