import { loadConfig, updateConfig, type AppConfig, type KeyState, type ModelState, type ModelCategory } from "./store";

export interface ChosenKey { key: KeyState }
export interface ChosenModel { model: ModelState }

const RATE_LIMIT_COOLDOWN_MS = 5 * 60 * 1000; // 5 min
const MODEL_COOLDOWN_MS = 60 * 1000; // 1 min

function isAvailableKey(k: KeyState, now: number): boolean {
  if (!k.enabled) return false;
  if (k.rateLimitedUntil && k.rateLimitedUntil > now) return false;
  return true;
}

function isAvailableModel(m: ModelState, now: number): boolean {
  if (!m.enabled) return false;
  if (m.failingUntil && m.failingUntil > now) return false;
  return true;
}

export async function pickKey(cfg: AppConfig): Promise<KeyState | null> {
  const now = Date.now();
  if (cfg.rotation.keyMode === "manual") {
    const k = cfg.keys.find((x) => x.id === cfg.rotation.activeKeyId);
    if (k && k.enabled) return k;
  }
  // Auto: prefer activeKeyId if available, else next available
  const ordered = [
    ...cfg.keys.filter((k) => k.id === cfg.rotation.activeKeyId),
    ...cfg.keys.filter((k) => k.id !== cfg.rotation.activeKeyId),
  ];
  for (const k of ordered) if (isAvailableKey(k, now)) return k;
  // Fallback: clear cooldowns and pick first enabled
  for (const k of cfg.keys) if (k.enabled) return k;
  return null;
}

function modelCats(m: ModelState): ModelCategory[] {
  return m.categories && m.categories.length > 0 ? m.categories : ["general"];
}

function categoryRank(m: ModelState, prefs: ModelCategory[]): number {
  const cats = modelCats(m);
  for (let i = 0; i < prefs.length; i++) {
    if (cats.includes(prefs[i]!)) return i;
  }
  return prefs.length + 1;
}

export async function pickModel(
  cfg: AppConfig,
  requestedName?: string,
  categoryPref?: ModelCategory[],
): Promise<ModelState | null> {
  const now = Date.now();
  if (cfg.rotation.modelMode === "manual") {
    const m = cfg.models.find((x) => x.id === cfg.rotation.activeModelId);
    if (m && m.enabled) return m;
  }
  // Auto: honor explicit name first
  if (requestedName) {
    const byName = cfg.models.find((m) => m.name === requestedName);
    if (byName && isAvailableModel(byName, now)) return byName;
  }
  // Vision is a hard requirement: never fall back to non-vision when vision is wanted
  const wantsVision = categoryPref?.[0] === "vision";
  if (wantsVision) {
    const visionModels = cfg.models.filter((m) => modelCats(m).includes("vision"));
    for (const m of visionModels) if (isAvailableModel(m, now)) return m;
    for (const m of visionModels) if (m.enabled) return m;
    // No vision model — fall through to normal picking; the request may still
    // partially work without images, better than throwing.
  }
  if (categoryPref && categoryPref.length > 0) {
    const sorted = [...cfg.models]
      .filter((m) => isAvailableModel(m, now))
      .sort((a, b) => categoryRank(a, categoryPref) - categoryRank(b, categoryPref));
    if (sorted.length > 0) return sorted[0]!;
  }
  // Fallback to legacy behavior: prefer active key, then any available
  const ordered = [
    ...cfg.models.filter((m) => m.id === cfg.rotation.activeModelId),
    ...cfg.models.filter((m) => m.id !== cfg.rotation.activeModelId),
  ];
  for (const m of ordered) if (isAvailableModel(m, now)) return m;
  for (const m of cfg.models) if (m.enabled) return m;
  return null;
}

export async function markKeyRateLimited(keyId: string, message: string): Promise<void> {
  await updateConfig((cfg) => {
    const k = cfg.keys.find((x) => x.id === keyId);
    if (!k) return;
    k.rateLimitedUntil = Date.now() + RATE_LIMIT_COOLDOWN_MS;
    k.lastError = `Rate limited: ${message}`;
    k.errorCount += 1;
    if (cfg.rotation.keyMode === "auto") {
      const next = pickNextEnabled(cfg.keys, keyId);
      if (next) cfg.rotation.activeKeyId = next.id;
    }
  });
}

export async function markKeyError(keyId: string, message: string): Promise<void> {
  await updateConfig((cfg) => {
    const k = cfg.keys.find((x) => x.id === keyId);
    if (!k) return;
    k.lastError = message;
    k.errorCount += 1;
  });
}

export async function markKeySuccess(keyId: string): Promise<void> {
  await updateConfig((cfg) => {
    const k = cfg.keys.find((x) => x.id === keyId);
    if (!k) return;
    k.successCount += 1;
    k.rateLimitedUntil = null;
    if (k.lastError && k.lastError.startsWith("Rate limited")) k.lastError = null;
  });
}

export async function markModelFailure(modelId: string, message: string): Promise<void> {
  await updateConfig((cfg) => {
    const m = cfg.models.find((x) => x.id === modelId);
    if (!m) return;
    m.failingUntil = Date.now() + MODEL_COOLDOWN_MS;
    m.lastError = message;
    m.errorCount += 1;
    if (cfg.rotation.modelMode === "auto") {
      const next = pickNextEnabled(cfg.models, modelId);
      if (next) cfg.rotation.activeModelId = next.id;
    }
  });
}

export async function markModelSuccess(modelId: string): Promise<void> {
  await updateConfig((cfg) => {
    const m = cfg.models.find((x) => x.id === modelId);
    if (!m) return;
    m.successCount += 1;
    m.failingUntil = null;
    m.lastError = null;
  });
}

function pickNextEnabled<T extends { id: string; enabled: boolean }>(items: T[], currentId: string): T | null {
  const idx = items.findIndex((x) => x.id === currentId);
  for (let i = 1; i <= items.length; i++) {
    const candidate = items[(idx + i) % items.length];
    if (candidate && candidate.enabled) return candidate;
  }
  return null;
}

export async function listAvailableKeysAndModels(): Promise<{ keys: KeyState[]; models: ModelState[] }> {
  const cfg = await loadConfig();
  const now = Date.now();
  return {
    keys: cfg.keys.filter((k) => isAvailableKey(k, now)),
    models: cfg.models.filter((m) => isAvailableModel(m, now)),
  };
}
