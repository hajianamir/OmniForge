import { defineConfig } from '@playwright/test';

// CI installs Playwright's Chromium. An existing Chromium can also be supplied
// explicitly, which is useful in restricted/offline development environments.
export default defineConfig({
  testDir: './tests/browser',
  timeout: 30_000,
  fullyParallel: true,
  reporter: 'list',
  use: {
    // Override to exercise the packaged SPA served by the API instead of Vite.
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:5173',
    browserName: 'chromium',
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
      args: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? ['--no-sandbox'] : [],
    },
  },
  webServer: process.env.PLAYWRIGHT_BASE_URL ? undefined : {
    command: 'npm run dev -- --host 0.0.0.0',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
