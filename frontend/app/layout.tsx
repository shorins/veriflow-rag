import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";

import "@/app/globals.css";
import { Providers } from "@/app/providers";

const manrope = Manrope({
  subsets: ["latin", "cyrillic"],
  variable: "--font-manrope",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin", "cyrillic"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
});

export const metadata: Metadata = {
  title: "trustRAG",
  description: "Grounded RAG with live claim verification and rewrite visualization",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru" className={`${manrope.variable} ${plexMono.variable}`}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}

