import { build } from "esbuild";
import { rm, chmod } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.resolve(here, "dist");

await rm(distDir, { recursive: true, force: true });

await build({
  entryPoints: [path.resolve(here, "src/cli.ts")],
  outfile: path.resolve(distDir, "nim.mjs"),
  platform: "node",
  target: "node18",
  bundle: true,
  format: "esm",
  logLevel: "info",
  loader: { ".html": "text" },
  banner: {
    js:
      "#!/usr/bin/env node\n" +
      "import { createRequire as __cr } from 'node:module';\n" +
      "import __p from 'node:path';\n" +
      "import __u from 'node:url';\n" +
      "globalThis.require = __cr(import.meta.url);\n" +
      "globalThis.__filename = __u.fileURLToPath(import.meta.url);\n" +
      "globalThis.__dirname = __p.dirname(globalThis.__filename);\n",
  },
});

await chmod(path.resolve(distDir, "nim.mjs"), 0o755);
console.log("Built dist/nim.mjs");
