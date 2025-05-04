export type UserRole = 'doctor' | 'client';

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  saveHistory: boolean;
}

export interface Message {
  id: string;
  text?: string;
  image?: string;
  audio?: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string, role: UserRole) => Promise<void>;
  signup: (name: string, email: string, password: string, role: UserRole, saveHistory: boolean) => Promise<void>;
  logout: () => void;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  sendMessage: (text?: string, image?: File, audio?: File) => Promise<void>;
  clearMessages: () => void;
}

export type InputMode = 'text' | 'image' | 'audio' | 'all';