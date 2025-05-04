# Multimodal Chatbot Application

A sophisticated full-stack chatbot application with text, image, and audio input capabilities.

## Features

- Multi-modal chat interface
- User authentication with role-based access
- Apple-inspired minimalist design
- React + Tailwind frontend
- Flask backend

## Setup

### Install Dependencies

```bash
# Install frontend dependencies
npm install

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Running the Application

Start both the frontend and backend:

```bash
npm run start:all
```

Or run them separately:

```bash
# Frontend
npm run dev

# Backend
npm run start:backend
```

The frontend will be available at http://localhost:5173 and the backend API at http://localhost:8001.

## Usage

### Authentication

For demo purposes, you can log in with these credentials:
- Doctor: doctor@example.com (any password)
- Client: client@example.com (any password)

### Chat Interface

- Use the floating action button (+) to select input mode:
  - Text: Type messages
  - Image: Upload images
  - Audio: Record audio messages

- Doctors have access to client chat history via the "Clients" tab
- Clients only have access to the chat interface

## Technical Details

- Frontend: React + TypeScript + Tailwind CSS + Framer Motion
- Backend: Flask with multipart/form-data support
- State Management: Zustand
- File Storage: Local storage in ./uploads directory