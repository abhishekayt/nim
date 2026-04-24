import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import type { ModelCategory } from "./store";

/**
 * Per-project overrides loaded from `.nimrc` (or `.nim-claude.json`) in the
 * directory where `nim` was launched. Lets you pin a model/category for a
 * specific project without touching global config.
 *
 * Example `.nimrc`:
 * ```json
 * {
 *   "category": "coding",
 *   "model": "qwen/qwen2.5-coder-32b-instruct",
 *   "cache": false,
 *   "thinking": true
 * }
 * ```
 */
export interface ProjectConfig {
  category?: ModelCategory;
  model?: string;
  provider?: string;
  cache?: boolean;
  thinking?: boolean;
  shortenToolDescriptions?: boolean;
  /**
   * If true, the cascade router skips the cheap-first attempt and goes
   * straight to the strongest tier for the chosen category.
   */
  preferAccuracy?: boolean;
  /**
   * Multi-sample voting for hard reasoning. Set to a number (2-5) to
   * always run that many parallel samples, or `true` to enable the
   * default heuristic (3 samples when category=reasoning-hard and
   * classifier confidence is below threshold).
   */
  multiSample?: boolean | number;
}

let cached: ProjectConfig | null = null;
let loadedFrom: string | null = null;

export function loadProjectConfig(cwd?: string): ProjectConfig {
  if (cached) return cached;
  const dir = cwd
    ?? process.env["NIM_PROJECT_DIR"]
    ?? process.cwd();
  const candidates = [".nimrc", ".nim-claude.json"];
  for (const name of candidates) {
    const p = path.join(dir, name);
    if (!existsSync(p)) continue;
    try {
      const raw = readFileSync(p, "utf8").trim();
      const parsed = raw.startsWith("{") ? JSON.parse(raw) : {};
      cached = parsed as ProjectConfig;
      loadedFrom = p;
      return cached;
    } catch {
      // ignore malformed file
    }
  }
  cached = {};
  return cached;
}

export function projectConfigPath(): string | null { return loadedFrom; }

export function resetProjectConfig(): void { cached = null; loadedFrom = null; }
