import React from 'react';
import { motion } from 'framer-motion';
import { AuthForm } from '../components/auth/AuthForm';
import { MessageCircle } from 'lucide-react';

export const Auth: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm py-4 px-6">
        <div className="max-w-7xl mx-auto flex items-center">
          <MessageCircle size={28} className="text-primary mr-2" />
          <h1 className="text-xl font-bold text-secondary-900">Multimodal Chat</h1>
        </div>
      </header>
      
      {/* Main content */}
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-center mb-8"
          >
            <h2 className="text-3xl font-bold text-secondary-900">Welcome to MedicalBot</h2>
            <p className="mt-2 text-secondary-600">
              The advanced AI assistant that understands text, images, and audio
            </p>
          </motion.div>
          
          <AuthForm />
        </div>
      </main>
      
      {/* Footer */}
      <footer className="py-6 text-center text-sm text-secondary-500">
        <p>© 2025 MedicalBot. All rights reserved.</p>
      </footer>
    </div>
  );
};