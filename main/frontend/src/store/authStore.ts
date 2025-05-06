import { create } from 'zustand';
import { AuthState, User, UserRole } from '../types';

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,

  login: async (email: string, password: string, role: UserRole) => {
    const response = await fetch('http://127.0.0.1:5000/signin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, role }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.message || 'Login failed');
    }

    const loggedInUser: User = {
      id: data.user_id,
      name: data.name,
      email: data.email,
      role: data.role,
      saveHistory: data.save_history,
    };

    set({ user: loggedInUser, isAuthenticated: true });
  },

  signup: async (name: string, email: string, password: string, role: UserRole, saveHistory: boolean) => {
    const response = await fetch('http://127.0.0.1:5000/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, role, save_history: saveHistory }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.message || 'Signup failed');
    }

    const newUser: User = {
      id: data.user_id,
      name: data.name,
      email: data.email,
      role: data.role,
      saveHistory: data.save_history,
    };

    set({ user: newUser, isAuthenticated: true });
  },

  logout: () => {
    set({ user: null, isAuthenticated: false });
  },
}));
