import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '../../utils/cn';

interface FABProps {
  icon: React.ReactNode;
  onClick: () => void;
  label?: string;
  isActive?: boolean;
  variant?: 'primary' | 'secondary' | 'accent';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const FloatingActionButton: React.FC<FABProps> = ({
  icon,
  onClick,
  label,
  isActive = false,
  variant = 'primary',
  size = 'md',
  className,
}) => {
  const variants = {
    primary: 'bg-primary text-white shadow-md hover:bg-primary-600',
    secondary: 'bg-secondary-100 text-secondary-900 shadow-md hover:bg-secondary-200',
    accent: 'bg-accent text-white shadow-md hover:bg-accent-600',
  };

  const sizes = {
    sm: 'w-10 h-10',
    md: 'w-12 h-12',
    lg: 'w-14 h-14',
  };

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={cn(
        'rounded-full flex items-center justify-center transition-all',
        variants[variant],
        sizes[size],
        isActive && 'ring-2 ring-offset-2 ring-primary',
        className
      )}
      onClick={onClick}
      aria-label={label}
    >
      {icon}
    </motion.button>
  );
};