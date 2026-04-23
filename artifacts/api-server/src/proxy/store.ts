import { promises as fs } from "node:fs";
import path from "node:path";
import os from "node:os";

const DATA_DIR =
  process.env["NIM_DATA_DIR"] ?? path.join(process.cwd(), ".data", "nim-proxy");
const CONFIG_FILE = path.join(DATA_DIR, "config.json");

export interface ProviderState {
  id: string;
  name: string;
  baseUrl: string;
  enabled: boolean;
}

export interface KeyState {
  id: string;
  label: string;
  key: string;
  enabled: boolean;
  rateLimitedUntil: number | null;
  lastError: string | null;
  successCount: number;
  errorCount: number;
  /** Which provider's baseUrl this key authenticates against. Defaults to "nim". */
  providerId?: string;
}

export const PROVIDER_PRESETS: Record<string, { name: string; baseUrl: string }> = {
  nim:        { name: "NVIDIA NIM",  baseUrl: "https://integrate.api.nvidia.com/v1" },
  groq:       { name: "Groq",         baseUrl: "https://api.groq.com/openai/v1" },
  openrouter: { name: "OpenRouter",   baseUrl: "https://openrouter.ai/api/v1" },
  together:   { name: "Together AI",  baseUrl: "https://api.together.xyz/v1" },
  deepinfra:  { name: "DeepInfra",    baseUrl: "https://api.deepinfra.com/v1/openai" },
};

export type ModelCategory = "coding" | "reasoning" | "vision" | "general";

export interface ModelState {
  id: string;
  name: string;
  enabled: boolean;
  failingUntil: number | null;
  lastError: string | null;
  successCount: number;
  errorCount: number;
  /**
   * Tags this model is best suited for. The router prefers models whose
   * categories match the classified request (e.g. coding, reasoning, vision).
   * Defaults to ["general"] if missing for backward compat.
   */
  categories?: ModelCategory[];
}

export interface RotationConfig {
  keyMode: "auto" | "manual";
  modelMode: "auto" | "manual";
  activeKeyId: string | null;
  activeModelId: string | null;
  nimBaseUrl: string;
}

export interface AppConfig {
  keys: KeyState[];
  models: ModelState[];
  rotation: RotationConfig;
  providers?: ProviderState[];
}

const DEFAULT_MODELS: Omit<
  ModelState,
  "successCount" | "errorCount" | "failingUntil" | "lastError"
>[] = [
  { id: "m1", name: "qwen/qwen2.5-coder-32b-instruct", enabled: true, categories: ["coding", "general"] },
  { id: "m2", name: "deepseek-ai/deepseek-r1", enabled: true, categories: ["reasoning", "coding"] },
  { id: "m3", name: "meta/llama-3.1-405b-instruct", enabled: true, categories: ["reasoning", "general"] },
  { id: "m4", name: "meta/llama-3.3-70b-instruct", enabled: true, categories: ["general", "coding"] },
  { id: "m5", name: "nvidia/llama-3.1-nemotron-70b-instruct", enabled: true, categories: ["general", "reasoning"] },
  { id: "m6", name: "mistralai/mixtral-8x22b-instruct-v0.1", enabled: true, categories: ["general"] },
  { id: "m7", name: "meta/llama-3.2-90b-vision-instruct", enabled: true, categories: ["vision", "general"] },
];

function defaultConfig(): AppConfig {
  const envKeys = [
    process.env["NIM_API_KEY_1"],
    process.env["NIM_API_KEY_2"],
    process.env["NIM_API_KEY_3"],
  ];
  const keys: KeyState[] = envKeys
    .map((k, i) =>
      k && k.trim().length > 0
        ? {
            id: `k${i + 1}`,
            label: `Key ${i + 1}`,
            key: k.trim(),
            enabled: true,
            rateLimitedUntil: null,
            lastError: null,
            successCount: 0,
            errorCount: 0,
          }
        : null,
    )
    .filter((k): k is KeyState => k !== null);

  const models: ModelState[] = DEFAULT_MODELS.map((m) => ({
    ...m,
    failingUntil: null,
    lastError: null,
    successCount: 0,
    errorCount: 0,
  }));

  return {
    keys: keys.map((k) => ({ ...k, providerId: "nim" })),
    models,
    rotation: {
      keyMode: "auto",
      modelMode: "auto",
      activeKeyId: keys[0]?.id ?? null,
      activeModelId: models[0]?.id ?? null,
      nimBaseUrl:
        process.env["NIM_BASE_URL"] ?? "https://integrate.api.nvidia.com/v1",
    },
    providers: defaultProviders(),
  };
}

function defaultProviders(): ProviderState[] {
  return [
    { id: "nim",        name: "NVIDIA NIM",  baseUrl: process.env["NIM_BASE_URL"] ?? PROVIDER_PRESETS["nim"]!.baseUrl, enabled: true },
    { id: "groq",       name: "Groq",        baseUrl: PROVIDER_PRESETS["groq"]!.baseUrl,        enabled: true },
    { id: "openrouter", name: "OpenRouter",  baseUrl: PROVIDER_PRESETS["openrouter"]!.baseUrl,  enabled: true },
  ];
}

export function resolveBaseUrl(cfg: AppConfig, key: KeyState): string {
  const providerId = key.providerId ?? "nim";
  const provider = (cfg.providers ?? []).find((p) => p.id === providerId);
  if (provider) return provider.baseUrl;
  return cfg.rotation.nimBaseUrl;
}

let cached: AppConfig | null = null;

export async function loadConfig(): Promise<AppConfig> {
  if (cached) return cached;
  try {
    const raw = await fs.readFile(CONFIG_FILE, "utf8");
    const parsed = JSON.parse(raw) as Partial<AppConfig>;
    cached = mergeWithDefaults(parsed);
  } catch {
    cached = defaultConfig();
    await saveConfig(cached);
  }
  return cached;
}

function mergeWithDefaults(loaded: Partial<AppConfig>): AppConfig {
  const def = defaultConfig();
  const keys = (loaded.keys && loaded.keys.length > 0 ? loaded.keys : def.keys)
    .map((k) => ({ ...k, providerId: k.providerId ?? "nim" }));
  const models =
    loaded.models && loaded.models.length > 0 ? loaded.models : def.models;
  const rotation: RotationConfig = {
    keyMode: loaded.rotation?.keyMode ?? "auto",
    modelMode: loaded.rotation?.modelMode ?? "auto",
    activeKeyId: loaded.rotation?.activeKeyId ?? keys[0]?.id ?? null,
    activeModelId: loaded.rotation?.activeModelId ?? models[0]?.id ?? null,
    nimBaseUrl: loaded.rotation?.nimBaseUrl ?? def.rotation.nimBaseUrl,
  };
  // Merge providers: keep user-defined, fill in any missing presets
  const providers = mergeProviders(loaded.providers ?? [], def.providers ?? []);
  return { keys, models, rotation, providers };
}

function mergeProviders(loaded: ProviderState[], defaults: ProviderState[]): ProviderState[] {
  const byId = new Map<string, ProviderState>();
  for (const p of defaults) byId.set(p.id, p);
  for (const p of loaded) byId.set(p.id, { ...byId.get(p.id), ...p });
  return Array.from(byId.values());
}

export async function saveConfig(cfg: AppConfig): Promise<void> {
  cached = cfg;
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.writeFile(CONFIG_FILE, JSON.stringify(cfg, null, 2), "utf8");
}

export async function updateConfig(
  mutator: (cfg: AppConfig) => void | Promise<void>,
): Promise<AppConfig> {
  const cfg = await loadConfig();
  await mutator(cfg);
  await saveConfig(cfg);
  return cfg;
}

export function maskKey(key: string): string {
  if (key.length <= 8) return "****";
  return `${key.slice(0, 4)}…${key.slice(-4)}`;
}

export function publicConfig(cfg: AppConfig) {
  return {
    keys: cfg.keys.map((k) => ({
      id: k.id,
      label: k.label,
      masked: maskKey(k.key),
      enabled: k.enabled,
      rateLimitedUntil: k.rateLimitedUntil,
      lastError: k.lastError,
      successCount: k.successCount,
      errorCount: k.errorCount,
      providerId: k.providerId ?? "nim",
    })),
    models: cfg.models.map((m) => ({ ...m })),
    rotation: { ...cfg.rotation },
    providers: (cfg.providers ?? []).map((p) => ({ ...p })),
  };
}

export function configFilePath(): string {
  return CONFIG_FILE;
}
