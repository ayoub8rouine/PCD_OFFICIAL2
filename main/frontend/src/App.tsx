import React from 'react';
import { Auth } from './pages/Auth';
import { Dashboard } from './pages/Dashboard';
import { useAuthStore } from './store/authStore';

function App() {
  const { isAuthenticated } = useAuthStore();
  
  return (
    <div className="min-h-screen bg-gray-50">
      {isAuthenticated ? <Dashboard /> : <Auth />}
    </div>
  );
}

export default App;