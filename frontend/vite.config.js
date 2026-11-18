import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server settings come from training-management-server/.env (see .env.example) or the environment.
export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, '..', 'STUDIO_'), ...process.env }
  const apiHost = env.STUDIO_HOST && env.STUDIO_HOST !== '0.0.0.0' ? env.STUDIO_HOST : '127.0.0.1'
  const api = env.STUDIO_API_URL || `http://${apiHost}:${env.STUDIO_PORT || 8700}`
  return {
    plugins: [react()],
    server: { port: Number(env.STUDIO_DEV_PORT || 5175), proxy: { '/api': api } },
  }
})
