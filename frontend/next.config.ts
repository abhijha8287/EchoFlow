import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // Pins the workspace root to this project -- without it, Next.js
  // auto-detects a stray package-lock.json elsewhere on this machine
  // (outside the repo) as a possible monorepo root and warns about it.
  outputFileTracingRoot: path.resolve(__dirname),
};

export default nextConfig;
