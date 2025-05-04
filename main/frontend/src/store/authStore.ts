import { create } from 'zustand';
import { AuthState, User, UserRole } from '../types';

// Mock users for demo purposes
const mockUsers: User[] = [
  {
    id: '1',
    name: 'Dr. Smith',
    email: 'doctor@example.com',
    role: 'doctor',
    saveHistory: true,
  },
  {
    id: '2',
    name: 'John Doe',
    email: 'client@example.com',
    role: 'client',
    saveHistory: true,
  },
];

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,
  
  login: async (email: string, password: string, role: UserRole) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Find user with matching email and role
    const user = mockUsers.find(u => u.email === email && u.role === role);
    
    if (user) {
      set({ user, isAuthenticated: true });
    } else {
      throw new Error('Invalid credentials');
    }
  },
  
  signup: async (name: string, email: string, password: string, role: UserRole, saveHistory: boolean) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Check if email is already in use
    if (mockUsers.some(u => u.email === email)) {
      throw new Error('Email already in use');
    }
    
    // Create new user
    const newUser: User = {
      id: Math.random().toString(36).substring(2, 9),
      name,
      email,
      role,
      saveHistory,
    };
    
    // Add to mock users (in a real app, this would be an API call)
    mockUsers.push(newUser);
    
    set({ user: newUser, isAuthenticated: true });
  },
  
  logout: () => {
    set({ user: null, isAuthenticated: false });
  },
}));