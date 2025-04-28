import { Message } from '../components/ChatMessage';

export const getWelcomeMessage = (): string => {
  return "Hello! I'm MediChat, your medical information assistant. I can provide general information about common health topics, but please remember that I'm not a replacement for professional medical advice.\n\nHow can I help you today?";
};

export const getBotResponse = async (query: string): Promise<string> => {
  try {
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error('Failed to get response from server');
    }

    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error('Error getting bot response:', error);
    throw new Error('Failed to get response from server');
  }
};