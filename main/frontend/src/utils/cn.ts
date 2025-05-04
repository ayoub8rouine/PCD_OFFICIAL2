import { ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Function to merge tailwind classes with clsx
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}