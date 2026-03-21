import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // @ts-ignore - resolve Next.js 16 HMR block
  allowedDevOrigins: ['127.0.0.1', 'localhost', '192.168.56.1'],
};

export default nextConfig;
