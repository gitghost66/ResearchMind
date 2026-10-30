import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies the API, so the browser talks to one origin and no
// CORS setup is needed. API_TARGET points it elsewhere (default: a local
// uvicorn on port 8000; the Docker stack is http://localhost:8080).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const target = env.API_TARGET || "http://localhost:8000";
  const proxy = Object.fromEntries(
    ["/api", "/health", "/ready", "/docs", "/openapi.json"].map((path) => [
      path,
      { target, changeOrigin: true },
    ]),
  );
  return {
    plugins: [react()],
    server: { port: 5173, proxy },
    preview: { port: 4173, proxy },
  };
});
