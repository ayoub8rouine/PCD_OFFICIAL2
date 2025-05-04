import { create } from 'zustand';
import axios from 'axios';
import { ChatState, Message } from '../types';

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,
  
  sendMessage: async (text?: string, image?: File, audio?: File) => {
    try {
      // Only send if at least one input is provided
      if (!text && !image && !audio) return;
      
      set({ isLoading: true });
      
      // Add user message to state immediately
      const userMessageId = Date.now().toString();
      const userMessage: Message = {
        id: userMessageId,
        text,
        image: image ? URL.createObjectURL(image) : undefined,
        audio: audio ? URL.createObjectURL(audio) : undefined,
        sender: 'user',
        timestamp: new Date(),
      };
      
      set(state => ({
        messages: [...state.messages, userMessage]
      }));
      
      // Prepare form data
      const formData = new FormData();
      if (text) formData.append('text', text);
      if (image) formData.append('image', image);
      if (audio) formData.append('audio', audio);
      
      // Send to backend
      const response = await axios.post('http://localhost:8001/api/chat', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Add bot response
      const botMessage: Message = {
        id: Date.now().toString(),
        text: response.data.text,
        image: response.data.image ? `http://localhost:8001/uploads/${response.data.image}` : undefined,
        audio: response.data.audio ? `http://localhost:8001/uploads/${response.data.audio}` : undefined,
        sender: 'bot',
        timestamp: new Date(),
      };
      
      set(state => ({
        messages: [...state.messages, botMessage],
        isLoading: false,
      }));
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: 'Sorry, there was an error processing your message. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };
      
      set(state => ({
        messages: [...state.messages, errorMessage],
        isLoading: false,
      }));
    }
  },
  
  clearMessages: () => {
    set({ messages: [] });
  },
}));