import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  publicDir: "../output",
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        essay: resolve(__dirname, "essay.html"),
      },
    },
  },
});
