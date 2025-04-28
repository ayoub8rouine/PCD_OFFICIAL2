import React, { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import Header from './components/Header';
import ChatContainer from './components/ChatContainer';
import ChatInput from './components/ChatInput';
import QuickActions from './components/QuickActions';
import LoginForm from './components/Auth/LoginForm';
import SignupForm from './components/Auth/SignupForm';
import DoctorDashboard from './components/DoctorDashboard';
import { Message } from './components/ChatMessage';
import { getBotResponse, getWelcomeMessage } from './utils/botResponses';
import { useAuthStore } from './store/authStore';
import { useChatStore } from './store/chatStore';

const mockClients = [
  { id: '1', email: 'patient1@example.com' },
  { id: '2', email: 'patient2@example.com' },
  { id: '3', email: 'patient3@example.com' },
];

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [showLogin, setShowLogin] = useState(true);
  const [doctorView, setDoctorView] = useState<'chat' | 'dashboard'>('dashboard');
  const user = useAuthStore((state) => state.user);
  const { addMessage } = useChatStore();

  useEffect(() => {
    if (user) {
      setTimeout(() => {
        const welcomeMessage: Message = {
          id: uuidv4(),
          text: getWelcomeMessage(),
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages([welcomeMessage]);
      }, 1000);
    }
  }, [user]);

  const simulateTyping = (text: string) => {
    const baseDelay = 1000;
    const charsPerSecond = 20;
    const calculatedDelay = Math.min(
      baseDelay + (text.length / charsPerSecond) * 1000,
      4000
    );
    
    return new Promise<void>(resolve => {
      setTimeout(() => {
        resolve();
      }, calculatedDelay);
    });
  };

  const handleSendMessage = async (text: string, image?: File) => {
    if (!text.trim() && !image) return;

    let imageUrl: string | undefined;
    if (image) {
      imageUrl = URL.createObjectURL(image);
    }

    const userMessage: Message = {
      id: uuidv4(),
      text,
      sender: 'user',
      timestamp: new Date(),
      imageUrl
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    if (user) {
      addMessage(user.id, {
        ...userMessage,
        userId: user.id
      });
    }
    
    setIsTyping(true);

    try {
      const botResponse = await getBotResponse(text);
      await simulateTyping(botResponse);

      const botMessage: Message = {
        id: uuidv4(),
        text: botResponse,
        sender: 'bot',
        timestamp: new Date()
      };
      
      setMessages(prevMessages => [...prevMessages, botMessage]);
      if (user) {
        addMessage(user.id, {
          ...botMessage,
          userId: user.id
        });
      }
    } catch (error) {
      const errorMessage: Message = {
        id: uuidv4(),
        text: "I apologize, but I'm having trouble connecting to the server. Please try again later.",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleQuickAction = (action: string) => {
    handleSendMessage(action);
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-100 flex flex-col">
        <Header />
        <div className="flex-1 flex items-center justify-center p-4">
          {showLogin ? (
            <LoginForm onToggleForm={() => setShowLogin(false)} />
          ) : (
            <SignupForm onToggleForm={() => setShowLogin(true)} />
          )}
        </div>
      </div>
    );
  }

  if (user.role === 'doctor') {
    return (
      <div className="flex flex-col h-screen bg-gray-100">
        <Header 
          onViewChange={setDoctorView} 
          currentView={doctorView}
        />
        <div className="flex-1 overflow-hidden">
          {doctorView === 'dashboard' ? (
            <DoctorDashboard clients={mockClients} />
          ) : (
            <>
              <ChatContainer messages={messages} isTyping={isTyping} />
              <QuickActions onSelectAction={handleQuickAction} disabled={isTyping} />
              <ChatInput onSendMessage={handleSendMessage} disabled={isTyping} />
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <Header />
      <ChatContainer messages={messages} isTyping={isTyping} />
      <QuickActions onSelectAction={handleQuickAction} disabled={isTyping} />
      <ChatInput onSendMessage={handleSendMessage} disabled={isTyping} />
    </div>
  );
}

export default App;