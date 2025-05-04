import React, { useState } from 'react';
import { useAuthStore } from '../../store/authStore';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { motion } from 'framer-motion';
import { Atom as At, KeyRound, User } from 'lucide-react';
import { UserRole } from '../../types';

type AuthMode = 'login' | 'signup';

export const AuthForm: React.FC = () => {
  const { login, signup } = useAuthStore();
  const [mode, setMode] = useState<AuthMode>('login');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Form state
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<UserRole>('client');
  const [saveHistory, setSaveHistory] = useState(true);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    
    try {
      if (mode === 'login') {
        await login(email, password, role);
      } else {
        await signup(name, email, password, role, saveHistory);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed');
    } finally {
      setIsLoading(false);
    }
  };
  
  const toggleMode = () => {
    setMode(mode === 'login' ? 'signup' : 'login');
    setError(null);
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="w-full max-w-md mx-auto bg-white p-8 rounded-xl shadow-lg"
    >
      <h2 className="text-2xl font-bold text-secondary-900 mb-6 text-center">
        {mode === 'login' ? 'Welcome back' : 'Create your account'}
      </h2>
      
      {error && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-error-lighter text-error-darker p-3 rounded-lg mb-4"
        >
          {error}
        </motion.div>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === 'signup' && (
          <Input
            label="Full Name"
            id="name"
            type="text"
            placeholder="John Doe"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            icon={<User size={18} className="text-secondary-400" />}
          />
        )}
        
        <Input
          label="Email"
          id="email"
          type="email"
          placeholder="you@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          icon={<At size={18} className="text-secondary-400" />}
        />
        
        <Input
          label="Password"
          id="password"
          type="password"
          placeholder="••••••••"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          icon={<KeyRound size={18} className="text-secondary-400" />}
        />
        
        <div>
          <label className="block text-sm font-medium text-secondary-700 mb-1">
            I am a
          </label>
          <div className="grid grid-cols-2 gap-3">
            <Button
              type="button"
              variant={role === 'client' ? 'primary' : 'outline'}
              onClick={() => setRole('client')}
              className="justify-center"
            >
              Client
            </Button>
            <Button
              type="button"
              variant={role === 'doctor' ? 'primary' : 'outline'}
              onClick={() => setRole('doctor')}
              className="justify-center"
            >
              Doctor
            </Button>
          </div>
        </div>
        
        {mode === 'signup' && (
          <div className="flex items-center">
            <input
              id="saveHistory"
              type="checkbox"
              checked={saveHistory}
              onChange={(e) => setSaveHistory(e.target.checked)}
              className="h-4 w-4 text-primary border-secondary-300 rounded focus:ring-primary"
            />
            <label htmlFor="saveHistory" className="ml-2 block text-sm text-secondary-700">
              Save my data history
            </label>
          </div>
        )}
        
        <Button
          type="submit"
          variant="primary"
          isLoading={isLoading}
          className="w-full"
        >
          {mode === 'login' ? 'Sign In' : 'Create Account'}
        </Button>
      </form>
      
      <div className="mt-6 text-center text-sm">
        <span className="text-secondary-600">
          {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
        </span>
        <button
          type="button"
          onClick={toggleMode}
          className="text-primary font-medium hover:underline focus:outline-none"
        >
          {mode === 'login' ? 'Sign up' : 'Sign in'}
        </button>
      </div>
      
      {/* Demo credentials */}
      {mode === 'login' && (
        <div className="mt-6 text-xs text-secondary-500 text-center">
          <p className="mb-1">Demo Credentials:</p>
          <p>Doctor: doctor@example.com</p>
          <p>Client: client@example.com</p>
          <p>(Any password will work)</p>
        </div>
      )}
    </motion.div>
  );
};