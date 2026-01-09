/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Enable class-based dark mode
  theme: {
    extend: {
      fontFamily: {
        sans: ['Josefin Sans', 'sans-serif'],
      },
      fontSize: {
        'xs': ['0.8125rem', { lineHeight: '1.25rem' }],     // 13px (was 12px)
        'sm': ['0.9375rem', { lineHeight: '1.375rem' }],    // 15px (was 14px)
        'base': ['1.0625rem', { lineHeight: '1.625rem' }],  // 17px (was 16px)
        'lg': ['1.1875rem', { lineHeight: '1.875rem' }],    // 19px (was 18px)
        'xl': ['1.3125rem', { lineHeight: '2rem' }],        // 21px (was 20px)
        '2xl': ['1.5625rem', { lineHeight: '2.25rem' }],    // 25px (was 24px)
        '3xl': ['1.9375rem', { lineHeight: '2.5rem' }],     // 31px (was 30px)
        '4xl': ['2.3125rem', { lineHeight: '2.75rem' }],    // 37px (was 36px)
      },
    },
  },
  plugins: [],
}
