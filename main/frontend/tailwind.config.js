/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0071E3',
          50: '#E0F0FF',
          100: '#B8DAFF',
          200: '#85B8FF',
          300: '#4D94FF',
          400: '#1A70FF',
          500: '#0071E3',
          600: '#0058B0',
          700: '#00417D',
          800: '#002A4A',
          900: '#001321',
        },
        secondary: {
          DEFAULT: '#8E8E93',
          50: '#F5F5F7',
          100: '#E5E5EA',
          200: '#D1D1D6',
          300: '#C7C7CC',
          400: '#AEAEB2',
          500: '#8E8E93',
          600: '#6C6C70',
          700: '#48484A',
          800: '#2C2C2E',
          900: '#1C1C1E',
        },
        accent: {
          DEFAULT: '#5E5CE6',
          50: '#EEEEFF',
          100: '#D1D0FF',
          200: '#B4B2FF',
          300: '#9795FF',
          400: '#7A78FF',
          500: '#5E5CE6',
          600: '#4A48B9',
          700: '#36348C',
          800: '#22215F',
          900: '#0F0E32',
        },
        success: {
          DEFAULT: '#34C759',
          lighter: '#DCFCE7',
          darker: '#166534',
        },
        warning: {
          DEFAULT: '#FF9500',
          lighter: '#FEF3C7',
          darker: '#92400E',
        },
        error: {
          DEFAULT: '#FF3B30',
          lighter: '#FEE2E2',
          darker: '#991B1B',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
      },
      spacing: {
        1: '0.25rem', // 4px
        2: '0.5rem',  // 8px
        3: '0.75rem', // 12px
        4: '1rem',    // 16px
        5: '1.25rem', // 20px
        6: '1.5rem',  // 24px
        8: '2rem',    // 32px
        10: '2.5rem', // 40px
        12: '3rem',   // 48px
      },
      borderRadius: {
        'sm': '0.25rem',
        DEFAULT: '0.5rem',
        'md': '0.75rem',
        'lg': '1rem',
        'xl': '1.5rem',
        '2xl': '2rem',
      },
      boxShadow: {
        'sm': '0 1px 3px rgba(0, 0, 0, 0.1)',
        DEFAULT: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
        'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      },
    },
  },
  plugins: [],
};