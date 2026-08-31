import { defineConfig } from "vite";

// GitHub Pages project sites are served from https://<user>.github.io/<repo>/,
// not the domain root, so every asset URL needs a "/<repo>/" prefix in that
// build. GITHUB_REPOSITORY ("owner/repo") is set automatically by GitHub
// Actions — read it here so the correct base is picked up with zero manual
// config, whatever the repo ends up being named. Falls back to "/" for local
// dev and any build run outside GitHub Actions.
const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1];
const base = repoName ? `/${repoName}/` : "/";

export default defineConfig({
  base,
  worker: {
    format: "es",
  },
  build: {
    target: "esnext",
  },
});
