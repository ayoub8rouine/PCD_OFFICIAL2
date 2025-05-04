import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStore } from '../../store/chatStore';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { FloatingActionButton } from '../ui/FloatingActionButton';
import { InputMode } from '../../types';
import { Mic, Image, Send, X, Keyboard, Plus } from 'lucide-react';

export const ChatInput: React.FC = () => {
  const { sendMessage, isLoading } = useChatStore();
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [showFabMenu, setShowFabMenu] = useState(false);
  const [text, setText] = useState('');
  const [image, setImage] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const audioRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  
  const toggleFabMenu = () => {
    setShowFabMenu(!showFabMenu);
  };
  
  const handleSendMessage = async () => {
    if (isLoading) return;
    
    // Only send if there's content
    if (!text && !image && !audio) return;
    
    await sendMessage(text, image || undefined, audio || undefined);
    
    // Reset state
    setText('');
    setImage(null);
    setImagePreview(null);
    setAudio(null);
    
    // Switch back to text mode if we sent a different type
    if (inputMode !== 'text' && inputMode !== 'all') {
      setInputMode('text');
    }
  };
  
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setImage(file);
    setImagePreview(URL.createObjectURL(file));
  };
  
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioRecorderRef.current = recorder;
      audioChunksRef.current = [];
      
      recorder.addEventListener('dataavailable', (e) => {
        audioChunksRef.current.push(e.data);
      });
      
      recorder.addEventListener('stop', () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
        setAudio(audioFile);
        stream.getTracks().forEach(track => track.stop());
      });
      
      recorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };
  
  const stopRecording = () => {
    if (audioRecorderRef.current && isRecording) {
      audioRecorderRef.current.stop();
      setIsRecording(false);
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  const setMode = (mode: InputMode) => {
    setInputMode(mode);
    setShowFabMenu(false);
  };
  
  const clearAttachments = () => {
    setImage(null);
    setImagePreview(null);
    setAudio(null);
    if (inputMode !== 'text' && inputMode !== 'all') {
      setInputMode('text');
    }
  };
  
  return (
    <div className="border-t border-secondary-200 bg-white p-4">
      {/* Attachments preview */}
      <AnimatePresence>
        {(imagePreview || audio) && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-3 flex flex-wrap gap-2"
          >
            {imagePreview && (
              <div className="relative">
                <img 
                  src={imagePreview} 
                  alt="Preview" 
                  className="h-20 w-auto rounded-md object-cover" 
                />
                <button
                  onClick={() => {
                    setImage(null);
                    setImagePreview(null);
                  }}
                  className="absolute -top-2 -right-2 bg-error text-white rounded-full p-1"
                >
                  <X size={16} />
                </button>
              </div>
            )}
            
            {audio && (
              <div className="relative">
                <audio src={URL.createObjectURL(audio)} controls className="h-10" />
                <button
                  onClick={() => setAudio(null)}
                  className="absolute -top-2 -right-2 bg-error text-white rounded-full p-1"
                >
                  <X size={16} />
                </button>
              </div>
            )}
            
            {(imagePreview || audio) && (
              <Button
                variant="outline"
                size="sm"
                onClick={clearAttachments}
                className="ml-auto"
              >
                Clear all
              </Button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Input area */}
      <div className="flex items-end gap-2">
        {/* Floating action button */}
        <div className="relative">
          <FloatingActionButton
            icon={<Plus size={24} />}
            onClick={toggleFabMenu}
            size="md"
            variant="secondary"
          />
          
          {/* FAB Menu */}
          <AnimatePresence>
            {showFabMenu && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="absolute bottom-16 left-0 flex flex-col gap-2"
              >
                <FloatingActionButton
                  icon={<Keyboard size={20} />}
                  onClick={() => setMode('text')}
                  isActive={inputMode === 'text'}
                  size="sm"
                  variant="primary"
                  label="Text input"
                />
                <FloatingActionButton
                  icon={<Image size={20} />}
                  onClick={() => {
                    setMode('image');
                    // Trigger file input click
                    setTimeout(() => imageInputRef.current?.click(), 100);
                  }}
                  isActive={inputMode === 'image'}
                  size="sm"
                  variant="primary"
                  label="Image input"
                />
                <FloatingActionButton
                  icon={<Mic size={20} />}
                  onClick={() => {
                    setMode('audio');
                    // Start recording
                    startRecording();
                  }}
                  isActive={inputMode === 'audio' || isRecording}
                  size="sm"
                  variant="primary"
                  label="Audio input"
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* Main input area */}
        <div className="flex-1">
          {inputMode === 'text' || inputMode === 'all' ? (
            <Input
              placeholder="Type your message..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
          ) : null}
          
          {inputMode === 'image' && (
            <div className="flex items-center bg-secondary-50 rounded-lg p-2">
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              <span className="text-secondary-600 text-sm">
                {image ? 'Image selected' : 'Click the image button to select an image'}
              </span>
            </div>
          )}
          
          {inputMode === 'audio' && (
            <div className="flex items-center bg-secondary-50 rounded-lg p-2">
              {isRecording ? (
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-error animate-pulse"></div>
                  <span className="text-secondary-600 text-sm">Recording... Click stop when finished</span>
                </div>
              ) : (
                <span className="text-secondary-600 text-sm">
                  {audio ? 'Audio recorded' : 'Click the microphone button to start recording'}
                </span>
              )}
            </div>
          )}
        </div>
        
        {/* Action buttons */}
        {inputMode === 'image' && !imagePreview ? (
          <Button
            onClick={() => imageInputRef.current?.click()}
            variant="primary"
          >
            <Image size={20} />
          </Button>
        ) : null}
        
        {inputMode === 'audio' && isRecording ? (
          <Button
            onClick={stopRecording}
            variant="danger"
          >
            <X size={20} />
          </Button>
        ) : null}
        
        <Button
          onClick={handleSendMessage}
          disabled={isLoading || (!text && !image && !audio)}
          variant="primary"
        >
          <Send size={20} />
        </Button>
      </div>
      
      {/* Hidden file inputs */}
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="hidden"
        ref={imageInputRef}
      />
    </div>
  );
};