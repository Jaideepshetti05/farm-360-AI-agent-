import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Stability settings
  reactStrictMode: true,
  // Allowed dev origins
  allowedDevOrigins: ['127.0.0.1', 'localhost', '192.168.56.1'],
  // Optimize for stability
  compiler: {
    // Disable styled-components compiler to reduce complexity
    styledComponents: false,
  },
};

export default nextConfig;
