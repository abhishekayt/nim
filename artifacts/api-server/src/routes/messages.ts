import { Router, type IRouter } from "express";
import { anthropicToOpenAI, augmentSystemForTools, openAIToAnthropic, type AnthropicRequest } from "../proxy/translator";
import { callNimNonStream, callNimStream } from "../proxy/nim-client";
import { streamOpenAIToAnthropic } from "../proxy/streaming";
import { classifyRequest } from "../proxy/classifier";
import { loadConfig, type ModelCategory } from "../proxy/store";
import { loadProjectConfig, projectConfigPath } from "../proxy/projectConfig";

const router: IRouter = Router();

let projectConfigLogged = false;

router.post("/v1/messages", async (req, res) => {
  const body = req.body as AnthropicRequest;
  if (!body || !Array.isArray(body.messages)) {
    res.status(400).json({
      type: "error",
      error: { type: "invalid_request_error", message: "messages array required" },
    });
    return;
  }
  const requestedModel = body.model || "claude-3-5-sonnet";
  const isStream = body.stream === true;

  try {
    const proj = loadProjectConfig();
    if (!projectConfigLogged) {
      projectConfigLogged = true;
      const path = projectConfigPath();
      if (path) console.log(`[nim] project config loaded from ${path}: ${JSON.stringify(proj)}`);
    }

    const classified = classifyRequest(body);
    // Project config can override the classifier's category choice.
    const categories: ModelCategory[] = proj.category
      ? [proj.category, ...classified.categories.filter((c) => c !== proj.category)]
      : classified.categories;
    const signals = proj.category
      ? [`.nimrc:${proj.category}`, ...classified.signals]
      : classified.signals;

    const augmented = augmentSystemForTools(body);
    const payload = anthropicToOpenAI(augmented, requestedModel);
    // Project config can pin a specific model, overriding rotation.
    if (proj.model) payload.model = proj.model;

    if (signals.length > 0) {
      console.log(`[nim] route → ${categories.join(">")}  (${signals.join(", ")})`);
    }

    const noCache = proj.cache === false;

    if (isStream) {
      const { ctx, body: upstream } = await callNimStream(payload, requestedModel, { categories, signals, noCache });
      console.log(`[nim] stream via ${ctx.modelName} (key ${ctx.keyId})`);
      await streamOpenAIToAnthropic({ upstream, res, requestedModel });
      return;
    }
    const { ctx, data } = await callNimNonStream(payload, requestedModel, { categories, signals, noCache });
    const cacheTag = ctx.keyId === "cache" ? " [cached]" : "";
    console.log(`[nim] reply via ${ctx.modelName} (key ${ctx.keyId})${cacheTag}`);
    res.json(openAIToAnthropic(data, requestedModel));
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[nim] /v1/messages failed:", msg);
    if (!res.headersSent) {
      res.status(502).json({ type: "error", error: { type: "api_error", message: msg } });
    } else {
      res.end();
    }
  }
});

router.get("/v1/models", async (_req, res) => {
  const cfg = await loadConfig();
  res.json({
    data: cfg.models.map((m) => ({
      id: m.name,
      object: "model",
      type: "model",
      display_name: m.name,
    })),
  });
});

export default router;
