import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  worker: { format: "es" },
  build: {
    target: "es2022",
    rollupOptions: {
      // the landing page, the game, the realism check and the methodology
      input: { main: resolve(__dirname, "index.html"), play: resolve(__dirname, "play.html"), realism: resolve(__dirname, "realism.html"),
               methodology: resolve(__dirname, "methodology.html") },
    },
  },
});
