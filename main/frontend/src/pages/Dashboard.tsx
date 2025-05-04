import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChatInterface } from '../components/chat/ChatInterface';
import { ClientHistory } from '../components/doctor/ClientHistory';
import { useAuthStore } from '../store/authStore';
import { Button } from '../components/ui/Button';
import { MessageCircle, Users } from 'lucide-react';

type TabType = 'chat' | 'clients';

export const Dashboard: React.FC = () => {
  const { user } = useAuthStore();
  const [activeTab, setActiveTab] = useState<TabType>('chat');
  
  // Only doctors can access client history
  const isDoctor = user?.role === 'doctor';
  
  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm py-4 px-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center">
            <MessageCircle size={28} className="text-primary mr-2" />
            <h1 className="text-xl font-bold text-secondary-900">MedicalBot</h1>
          </div>
          
          {isDoctor && (
            <div className="flex space-x-2">
              <Button
                variant={activeTab === 'chat' ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setActiveTab('chat')}
                leftIcon={<MessageCircle size={16} />}
              >
                Chat
              </Button>
              <Button
                variant={activeTab === 'clients' ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setActiveTab('clients')}
                leftIcon={<Users size={16} />}
              >
                Clients
              </Button>
            </div>
          )}
        </div>
      </header>
      
      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        <div className="h-full max-w-7xl mx-auto bg-gray-50">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="h-full"
          >
            {activeTab === 'chat' ? (
              <ChatInterface />
            ) : (
              <ClientHistory />
            )}
          </motion.div>
        </div>
      </main>
    </div>
  );
};