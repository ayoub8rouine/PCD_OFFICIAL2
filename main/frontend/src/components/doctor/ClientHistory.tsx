import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { MessageBubble } from '../chat/MessageBubble';
import { Message } from '../../types';

// Mock data for client history
const mockClients = [
  {
    id: '1',
    name: 'John Doe',
    email: 'john@example.com',
    lastActive: new Date(2025, 3, 15),
    messages: [
      {
        id: '101',
        text: 'Hi, I have a question about my treatment',
        sender: 'user',
        timestamp: new Date(2025, 3, 14, 10, 30)
      },
      {
        id: '102',
        text: 'Hello John, what would you like to know?',
        sender: 'bot',
        timestamp: new Date(2025, 3, 14, 10, 32)
      },
      {
        id: '103',
        text: 'Is it normal to feel tired after taking the medication?',
        sender: 'user',
        timestamp: new Date(2025, 3, 14, 10, 33)
      },
      {
        id: '104',
        text: 'Yes, fatigue can be a common side effect. If it persists for more than a few days, please let me know.',
        sender: 'bot',
        timestamp: new Date(2025, 3, 14, 10, 35)
      }
    ] as Message[]
  },
  {
    id: '2',
    name: 'Jane Smith',
    email: 'jane@example.com',
    lastActive: new Date(2025, 3, 14),
    messages: [
      {
        id: '201',
        text: 'Hello, I need to reschedule my appointment',
        sender: 'user',
        timestamp: new Date(2025, 3, 13, 9, 15)
      },
      {
        id: '202',
        text: 'Hi Jane, I can help with that. When would you like to reschedule?',
        sender: 'bot',
        timestamp: new Date(2025, 3, 13, 9, 17)
      },
      {
        id: '203',
        image: 'https://images.pexels.com/photos/3938022/pexels-photo-3938022.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
        sender: 'user',
        timestamp: new Date(2025, 3, 13, 9, 18)
      }
    ] as Message[]
  }
];

export const ClientHistory: React.FC = () => {
  const [selectedClient, setSelectedClient] = useState(mockClients[0]);
  
  return (
    <div className="flex h-full">
      {/* Client list sidebar */}
      <div className="w-1/3 border-r border-secondary-200 bg-white overflow-y-auto">
        <div className="p-4 border-b border-secondary-200">
          <h2 className="text-lg font-semibold text-secondary-900">Clients</h2>
          <p className="text-sm text-secondary-500">{mockClients.length} total</p>
        </div>
        
        <div className="divide-y divide-secondary-100">
          {mockClients.map((client) => (
            <motion.button
              key={client.id}
              whileHover={{ backgroundColor: 'rgba(0,0,0,0.02)' }}
              onClick={() => setSelectedClient(client)}
              className={`w-full p-4 text-left ${selectedClient.id === client.id ? 'bg-secondary-50' : ''}`}
            >
              <div className="flex justify-between">
                <h3 className="font-medium text-secondary-900">{client.name}</h3>
                <span className="text-xs text-secondary-500">
                  {new Date(client.lastActive).toLocaleDateString()}
                </span>
              </div>
              <p className="text-sm text-secondary-600 truncate mt-1">{client.email}</p>
              <p className="text-xs text-secondary-500 mt-1">
                {client.messages.length} messages
              </p>
            </motion.button>
          ))}
        </div>
      </div>
      
      {/* Chat history */}
      <div className="flex-1 flex flex-col">
        <div className="bg-white border-b border-secondary-200 p-4">
          <h2 className="text-lg font-semibold text-secondary-900">{selectedClient.name}</h2>
          <p className="text-sm text-secondary-500">{selectedClient.email}</p>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 flex flex-col">
          {selectedClient.messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </div>
      </div>
    </div>
  );
};