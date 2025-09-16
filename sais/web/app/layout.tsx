import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SAIS - Saúde Inteligente",
  description: "Portal unificado para agendamentos e comunicação inteligente.",
  metadataBase: new URL("https://example.com")
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body className="min-h-screen bg-slate-50 text-slate-900">
        {children}
      </body>
    </html>
  );
}
