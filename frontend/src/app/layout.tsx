import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Farm360 AI — Intelligent Agricultural Advisor",
  description:
    "Expert AI-powered agricultural advisor for Indian farmers. Get instant advice on crops, diseases, soil, irrigation, and livestock.",
  keywords:
    "farming AI, crop advice, agricultural advisor, India farming, crop disease, irrigation, dairy, livestock",
  authors: [{ name: "Farm360 AI" }],
  robots: "index, follow",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#0a0a0a",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full flex flex-col">{children}</body>
    </html>
  );
}
