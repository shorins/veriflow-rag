import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  typedRoutes: false,
  outputFileTracingRoot: path.join(__dirname, ".."),
};

export default nextConfig;
