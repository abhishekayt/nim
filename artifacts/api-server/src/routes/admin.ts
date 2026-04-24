import { Router, type IRouter } from "express";
import { loadConfig, updateConfig, publicConfig, PROVIDER_PRESETS } from "../proxy/store";
import { getTelemetryDb, clearTelemetryDb, updateModelRankings, getModelRankings } from "../proxy/telemetryDb";
import { cacheClear, cacheStats } from "../proxy/cache";
import { getCacheStats, clearConversationCache } from "../proxy/conversationCache";

const router: IRouter = Router();

router.get("/admin/telemetry", async (_req, res) => {
  const telemetry = await getTelemetryDb(100);
  res.json({ ...telemetry, cache: cacheStats(), conversationCache: await getCacheStats() });
});

router.post("/admin/telemetry/clear", async (_req, res) => {
  await clearTelemetryDb();
  res.json({ ok: true });
});

router.post("/admin/cache/clear", (_req, res) => {
  cacheClear();
  res.json({ ok: true });
});

router.post("/admin/conversation-cache/clear", async (_req, res) => {
  await clearConversationCache();
  res.json({ ok: true });
});

router.get("/admin/providers", (_req, res) => {
  res.json({ presets: PROVIDER_PRESETS });
});

router.get("/admin/status", async (_req, res) => {
  const cfg = await loadConfig();
  res.json(publicConfig(cfg));
});

router.get("/admin/rankings", async (req, res) => {
  const category = typeof req.query.category === "string" ? req.query.category : undefined;
  const rankings = await getModelRankings(category);
  res.json({ rankings });
});

router.post("/admin/rankings/update", async (_req, res) => {
  await updateModelRankings();
  const rankings = await getModelRankings();
  res.json({ ok: true, rankings });
});

router.post("/admin/keys", async (req, res) => {
  const { label, key, providerId } = req.body as { label?: string; key?: string; providerId?: string };
  if (!key || typeof key !== "string" || key.trim().length < 8) {
    res.status(400).json({ error: "key must be a non-empty string" });
    return;
  }
  const cfg = await updateConfig((c) => {
    if (c.keys.length >= 3) throw new Error("Maximum 3 keys allowed");
    const id = `k${Date.now().toString(36)}`;
    const provId = (providerId && PROVIDER_PRESETS[providerId]) ? providerId : "nim";
    c.keys.push({
      id,
      label: label || `Key ${c.keys.length + 1}`,
      key: key.trim(),
      enabled: true,
      rateLimitedUntil: null,
      lastError: null,
      successCount: 0,
      errorCount: 0,
      providerId: provId,
    });
    if (!c.rotation.activeKeyId) c.rotation.activeKeyId = id;
  }).catch((e: unknown) => { throw e; });
  res.json(publicConfig(cfg));
});

router.patch("/admin/keys/:id", async (req, res) => {
  const { id } = req.params;
  const { label, enabled, key, providerId } = req.body as { label?: string; enabled?: boolean; key?: string; providerId?: string };
  const cfg = await updateConfig((c) => {
    const k = c.keys.find((x) => x.id === id);
    if (!k) throw new Error("Key not found");
    if (typeof label === "string") k.label = label;
    if (typeof enabled === "boolean") k.enabled = enabled;
    if (typeof key === "string" && key.trim().length >= 8) k.key = key.trim();
    if (typeof providerId === "string" && PROVIDER_PRESETS[providerId]) k.providerId = providerId;
  });
  res.json(publicConfig(cfg));
});

router.delete("/admin/keys/:id", async (req, res) => {
  const { id } = req.params;
  const cfg = await updateConfig((c) => {
    c.keys = c.keys.filter((x) => x.id !== id);
    if (c.rotation.activeKeyId === id) c.rotation.activeKeyId = c.keys[0]?.id ?? null;
  });
  res.json(publicConfig(cfg));
});

router.post("/admin/keys/:id/clear-cooldown", async (req, res) => {
  const { id } = req.params;
  const cfg = await updateConfig((c) => {
    const k = c.keys.find((x) => x.id === id);
    if (k) { k.rateLimitedUntil = null; k.lastError = null; }
  });
  res.json(publicConfig(cfg));
});

router.post("/admin/models", async (req, res) => {
  const { name } = req.body as { name?: string };
  if (!name || typeof name !== "string") {
    res.status(400).json({ error: "name required" });
    return;
  }
  const cfg = await updateConfig((c) => {
    const id = `m${Date.now().toString(36)}`;
    c.models.push({ id, name: name.trim(), enabled: true, failingUntil: null, lastError: null, successCount: 0, errorCount: 0 });
  });
  res.json(publicConfig(cfg));
});

router.patch("/admin/models/:id", async (req, res) => {
  const { id } = req.params;
  const { enabled, name } = req.body as { enabled?: boolean; name?: string };
  const cfg = await updateConfig((c) => {
    const m = c.models.find((x) => x.id === id);
    if (!m) throw new Error("Model not found");
    if (typeof enabled === "boolean") m.enabled = enabled;
    if (typeof name === "string") m.name = name.trim();
  });
  res.json(publicConfig(cfg));
});

router.delete("/admin/models/:id", async (req, res) => {
  const { id } = req.params;
  const cfg = await updateConfig((c) => {
    c.models = c.models.filter((x) => x.id !== id);
    if (c.rotation.activeModelId === id) c.rotation.activeModelId = c.models[0]?.id ?? null;
  });
  res.json(publicConfig(cfg));
});

router.post("/admin/models/:id/clear-cooldown", async (req, res) => {
  const { id } = req.params;
  const cfg = await updateConfig((c) => {
    const m = c.models.find((x) => x.id === id);
    if (m) { m.failingUntil = null; m.lastError = null; }
  });
  res.json(publicConfig(cfg));
});

router.put("/admin/rotation", async (req, res) => {
  const { keyMode, modelMode, activeKeyId, activeModelId, nimBaseUrl } = req.body as Partial<{
    keyMode: "auto" | "manual"; modelMode: "auto" | "manual";
    activeKeyId: string | null; activeModelId: string | null;
    nimBaseUrl: string;
  }>;
  const cfg = await updateConfig((c) => {
    if (keyMode === "auto" || keyMode === "manual") c.rotation.keyMode = keyMode;
    if (modelMode === "auto" || modelMode === "manual") c.rotation.modelMode = modelMode;
    if (activeKeyId === null || (typeof activeKeyId === "string" && c.keys.some((k) => k.id === activeKeyId))) {
      c.rotation.activeKeyId = activeKeyId;
    }
    if (activeModelId === null || (typeof activeModelId === "string" && c.models.some((m) => m.id === activeModelId))) {
      c.rotation.activeModelId = activeModelId;
    }
    if (typeof nimBaseUrl === "string" && nimBaseUrl.startsWith("http")) c.rotation.nimBaseUrl = nimBaseUrl.trim();
  });
  res.json(publicConfig(cfg));
});

export default router;
