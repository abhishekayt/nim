import { Router, type IRouter } from "express";
import { anthropicToOpenAI, augmentSystemForTools, openAIToAnthropic, buildModelSystemPrompt, type AnthropicRequest } from "../proxy/translator";
import { callNimNonStream, callNimStream } from "../proxy/nim-client";
import { streamOpenAIToAnthropic } from "../proxy/streaming";
import { classifyRequestWithConfidence } from "../proxy/classifier";
import { loadConfig, type ModelCategory } from "../proxy/store";
import { loadProjectConfig, projectConfigPath } from "../proxy/projectConfig";
import { getContextWindow } from "../proxy/systemPrompts";
import { summarizeConversation } from "../proxy/summarizer";
import { augmentRequestWithFileContext } from "../proxy/requestAugmenter";
import { appendToConversationCache } from "../proxy/conversationCache";

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

    const classified = classifyRequestWithConfidence(body);
    // Project config can override the classifier's category choice.
    const categories: ModelCategory[] = proj.category
      ? [proj.category, ...classified.categories.filter((c) => c !== proj.category)]
      : classified.categories;
    const signals = proj.category
      ? [`.nimrc:${proj.category}`, ...classified.signals]
      : classified.signals;

    // 1. Apply request augmentation (file context injection)
    let messages = body.messages;
    let systemPrompt = typeof body.system === "string" ? body.system : null;
    const augmentation = augmentRequestWithFileContext(
      messages,
      systemPrompt,
      { projectDir: process.env["NIM_PROJECT_DIR"] || process.cwd(), injectDirectoryTree: classified.signals.includes("coding-hints") || classified.signals.includes("coding-tools") }
    );
    if (augmentation.systemAddendum) {
      systemPrompt = systemPrompt ? `${systemPrompt}\n\n${augmentation.systemAddendum}` : augmentation.systemAddendum;
    }
    if (augmentation.modifiedMessages.length > 0) {
      messages = augmentation.modifiedMessages as AnthropicRequest["messages"];
    }

    // 2. Apply context window summarization
    const ctxWindow = getContextWindow(requestedModel);
    const summarized = summarizeConversation(messages, ctxWindow);
    if (summarized.wasSummarized) {
      messages = summarized.messages;
      signals.push("summarized");
      console.log(`[nim] summarized conversation from ${summarized.originalCount} to ${messages.length} messages`);
    }

    // 3. Build per-model system prompt
    const cfg = await loadConfig();
    const targetModel = await (async () => {
      if (proj.model) return proj.model;
      const m = categories.length > 0 ? cfg.models.find((m) => m.categories?.includes(categories[0]!)) : cfg.models[0];
      return m?.name ?? requestedModel;
    })();

    const hasTools = !!(body.tools && body.tools.length > 0);
    const enableThinking = proj.thinking !== false;
    const builtSystem = buildModelSystemPrompt(systemPrompt, targetModel, hasTools, enableThinking);

    const augmentedReq: AnthropicRequest = {
      ...body,
      messages,
      system: builtSystem,
    };

    const augmented = augmentSystemForTools(augmentedReq);
    const payload = anthropicToOpenAI(augmented, requestedModel);
    // Project config can pin a specific model, overriding rotation.
    if (proj.model) payload.model = proj.model;

    if (signals.length > 0) {
      console.log(`[nim] route → ${categories.join(">")} (confidence: ${(classified.confidence * 100).toFixed(0)}%) (${signals.join(", ")})`);
    }

    const noCache = proj.cache === false;

    if (isStream) {
      const { ctx, body: upstream } = await callNimStream(payload, requestedModel, { categories, signals, noCache });
      console.log(`[nim] stream via ${ctx.modelName} (key ${ctx.keyId})`);
      await streamOpenAIToAnthropic({ upstream, res, req, requestedModel });
      return;
    }
    const { ctx, data } = await callNimNonStream(payload, requestedModel, { categories, signals, noCache });
    const cacheTag = ctx.keyId === "cache" ? " [cached]" : "";
    console.log(`[nim] reply via ${ctx.modelName} (key ${ctx.keyId})${cacheTag}`);

    // Store in conversation cache
    const responseContent = data.choices?.[0]?.message?.content ?? "";
    await appendToConversationCache(
      { model: requestedModel, messages: body.messages },
      responseContent,
      ctx.modelName,
      categories[0] ?? null,
      data.usage?.completion_tokens ?? 0,
    );

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
