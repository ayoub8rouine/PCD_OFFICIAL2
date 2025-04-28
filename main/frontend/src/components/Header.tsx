import React from 'react';
import { Stethoscope, LogOut, MessageSquare, Users } from 'lucide-react';
import { useAuthStore } from '../store/authStore';

interface HeaderProps {
  onViewChange?: (view: 'chat' | 'dashboard') => void;
  currentView?: 'chat' | 'dashboard';
}

const Header: React.FC<HeaderProps> = ({ onViewChange, currentView }) => {
  const { user, logout } = useAuthStore();

  return (
    <header className="bg-gradient-to-r from-blue-600 to-blue-500 text-white p-4 shadow-md">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Stethoscope size={28} className="text-white" />
          <h1 className="text-xl font-bold">MediChat</h1>
        </div>
        <div className="flex items-center gap-4">
          {user?.role === 'doctor' && onViewChange && (
            <div className="flex items-center gap-2 mr-4">
              <button
                onClick={() => onViewChange('dashboard')}
                className={`flex items-center gap-1 px-3 py-1.5 rounded-lg transition-colors ${
                  currentView === 'dashboard'
                    ? 'bg-white text-blue-600'
                    : 'hover:bg-blue-500'
                }`}
              >
                <Users size={18} />
                <span className="hidden md:inline">Patients</span>
              </button>
              <button
                onClick={() => onViewChange('chat')}
                className={`flex items-center gap-1 px-3 py-1.5 rounded-lg transition-colors ${
                  currentView === 'chat'
                    ? 'bg-white text-blue-600'
                    : 'hover:bg-blue-500'
                }`}
              >
                <MessageSquare size={18} />
                <span className="hidden md:inline">Chat</span>
              </button>
            </div>
          )}
          {user && (
            <>
              <span className="text-sm">
                {user.email} ({user.role})
              </span>
              <button
                onClick={logout}
                className="flex items-center gap-1 hover:text-blue-200 transition-colors"
              >
                <LogOut size={18} />
                <span className="hidden md:inline">Logout</span>
              </button>
            </>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;