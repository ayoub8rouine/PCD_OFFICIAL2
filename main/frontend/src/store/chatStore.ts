import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';

export interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  imageUrl?: string;
  userId: string;
}

interface ChatState {
  conversations: Record<string, ChatMessage[]>;
  addMessage: (userId: string, message: ChatMessage) => void;
  getConversation: (userId: string) => ChatMessage[];
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: {},
  addMessage: (userId, message) => {
    set((state) => ({
      conversations: {
        ...state.conversations,
        [userId]: [...(state.conversations[userId] || []), message],
      },
    }));
  },
  getConversation: (userId) => {
    return get().conversations[userId] || [];
  },
}));