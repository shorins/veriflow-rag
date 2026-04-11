import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#fcfcfb",
        panel: "#ffffff",
        ink: "#18181b",
        mutedink: "#71717a",
        accent: "#0f766e",
        warning: "#f59e0b",
        danger: "#dc2626",
      },
      fontFamily: {
        sans: ["var(--font-manrope)", "sans-serif"],
        mono: ["var(--font-plex-mono)", "monospace"],
      },
      boxShadow: {
        panel: "0 10px 30px rgba(24, 24, 27, 0.06)",
      },
    },
  },
  plugins: [],
};

export default config;

