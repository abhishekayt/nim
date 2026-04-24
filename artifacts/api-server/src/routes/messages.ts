import { Router, type IRouter } from "express";
import {
  anthropicToOpenAI,
  augmentSystemForTools,
  openAIToAnthropic,
  buildModelSystemPrompt,
  type AnthropicRequest,
} from "../proxy/translator";
import { callNimNonStream, callNimStream, type CallOptions } from "../proxy/nim-client";
import { streamOpenAIToAnthropic } from "../proxy/streaming";
import { classifyRequestWithConfidence } from "../proxy/classifier";
import { loadConfig, type ModelCategory } from "../proxy/store";
import { loadProjectConfig, projectConfigPath } from "../proxy/projectConfig";
import { getContextWindow } from "../proxy/systemPrompts";
import { summarizeConversation } from "../proxy/summarizer";
import { augmentRequestWithFileContext } from "../proxy/requestAugmenter";
import { appendToConversationCache } from "../proxy/conversationCache";
import {
  extractCacheBreakpoints,
  pickBestBreakpoint,
  prefixHash,
  stripCacheControl,
} from "../proxy/promptCache";
import { buildCascadePlan } from "../proxy/cascade";
import {
  runVerifier,
  recordVerifierRun,
  formatDiagnosticsForModel,
} from "../proxy/verifier";
import { runMultiSample } from "../proxy/multiSample";

const router: IRouter = Router();

let projectConfigLogged = false;

const MULTI_SAMPLE_DEFAULT_N = 3;
const MULTI_SAMPLE_CONFIDENCE_THRESHOLD = 0.55;

const VERIFIER_ON_EDIT =
  (process.env["NIM_VERIFY_ON_EDIT"] ?? "").toLowerCase() === "1" ||
  (process.env["NIM_VERIFY_ON_EDIT"] ?? "").toLowerCase() === "true";

const EDIT_TOOL_NAMES = new Set(["Edit", "Write", "MultiEdit", "Apply", "Patch"]);

function detectImages(req: AnthropicRequest): boolean {
  for (const m of req.messages ?? []) {
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if ((block as { type?: string }).type === "image") return true;
      }
    }
  }
  return false;
}

function extractTouchedFiles(
  toolCalls: ReadonlyArray<{ function: { name: string; arguments: string } }>,
): string[] {
  const files = new Set<string>();
  for (const tc of toolCalls) {
    if (!EDIT_TOOL_NAMES.has(tc.function.name)) continue;
    try {
      const args = JSON.parse(tc.function.arguments || "{}") as {
        file_path?: unknown;
        path?: unknown;
        files?: unknown;
      };
      const f = (typeof args.file_path === "string" && args.file_path)
        || (typeof args.path === "string" && args.path);
      if (f) files.add(f);
      if (Array.isArray(args.files)) {
        for (const entry of args.files) {
          const p = (entry as { file_path?: unknown; path?: unknown })?.file_path
            ?? (entry as { file_path?: unknown; path?: unknown })?.path;
          if (typeof p === "string") files.add(p);
        }
      }
    } catch {
      /* ignore unparseable args */
    }
  }
  return [...files];
}

router.post("/v1/messages", async (req, res) => {
  // Strip cache_control markers before any further processing — the
  // OpenAI-compatible NIM API does not understand them. We extract the
  // breakpoint info first so we can use it for cache-key telemetry.
  const rawBody = req.body as AnthropicRequest;
  if (!rawBody || !Array.isArray(rawBody.messages)) {
    res.status(400).json({
      type: "error",
      error: { type: "invalid_request_error", message: "messages array required" },
    });
    return;
  }

  const breakpoints = extractCacheBreakpoints(rawBody);
  const bestBreakpoint = pickBestBreakpoint(rawBody);
  const promptCacheHash = bestBreakpoint
    ? prefixHash(rawBody, bestBreakpoint)
    : null;

  const body = stripCacheControl(rawBody);
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

    if (breakpoints.length > 0) {
      signals.push(`prompt-cache:${breakpoints.length}bp`);
    }
    if (promptCacheHash) {
      signals.push(`prefix:${promptCacheHash.slice(0, 8)}`);
    }

    // 1. Apply request augmentation (file context injection)
    let messages = body.messages;
    let systemPrompt = typeof body.system === "string" ? body.system : null;
    const augmentation = augmentRequestWithFileContext(
      messages,
      systemPrompt,
      {
        projectDir: process.env["NIM_PROJECT_DIR"] || process.cwd(),
        injectDirectoryTree:
          classified.signals.includes("coding-hints") ||
          classified.signals.includes("coding-tools"),
      },
    );
    if (augmentation.systemAddendum) {
      systemPrompt = systemPrompt
        ? `${systemPrompt}\n\n${augmentation.systemAddendum}`
        : augmentation.systemAddendum;
    }
    if (augmentation.modifiedMessages.length > 0) {
      messages = augmentation.modifiedMessages as AnthropicRequest["messages"];
    }
    if (augmentation.injectedFiles.length > 0) {
      signals.push(`augment:${augmentation.injectedFiles.length}files`);
    }

    // 2. Apply context window summarization
    const ctxWindow = getContextWindow(requestedModel);
    const summarized = summarizeConversation(messages, ctxWindow);
    if (summarized.wasSummarized) {
      messages = summarized.messages;
      signals.push(
        `summarized:${summarized.details.droppedCount}dropped/${summarized.details.dedupedReads}deduped`,
      );
      console.log(
        `[nim] summarized conversation from ${summarized.originalCount} to ${messages.length} messages (${summarized.details.dedupedReads} dedup'd reads)`,
      );
    }

    // 3. Build per-model system prompt
    const cfg = await loadConfig();
    const targetModel = await (async () => {
      if (proj.model) return proj.model;
      const m =
        categories.length > 0
          ? cfg.models.find((m) => m.categories?.includes(categories[0]!))
          : cfg.models[0];
      return m?.name ?? requestedModel;
    })();

    const hasTools = !!(body.tools && body.tools.length > 0);
    const hasImages = detectImages(body);
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

    // 4. Build cascade routing plan.
    const cascadePlan = buildCascadePlan({
      categories,
      confidence: classified.confidence,
      hasTools,
      hasImages,
      preferAccuracy: proj.preferAccuracy === true,
    });
    if (cascadePlan.startedCheap) signals.push(`cascade:cheap-first`);
    signals.push(`cascade:${cascadePlan.tiers.join(">")}`);

    if (signals.length > 0) {
      console.log(
        `[nim] route → ${categories.join(">")} (confidence: ${(classified.confidence * 100).toFixed(0)}%) (${signals.join(", ")})`,
      );
    }

    const noCache = proj.cache === false;
    const callOpts: CallOptions = {
      categories,
      signals,
      noCache,
      cascadePlan,
    };

    if (isStream) {
      const { ctx, body: upstream } = await callNimStream(payload, requestedModel, callOpts);
      console.log(
        `[nim] stream via ${ctx.modelName} (key ${ctx.keyId})${ctx.cacheLayer ? ` [${ctx.cacheLayer}-cache]` : ""}`,
      );
      await streamOpenAIToAnthropic({ upstream, res, req, requestedModel });
      return;
    }

    // 5. Multi-sample path for hard reasoning under low confidence
    //    (only when explicitly enabled; non-stream only).
    const multiN =
      typeof proj.multiSample === "number" && proj.multiSample > 1
        ? Math.min(5, proj.multiSample)
        : (categories[0] === "reasoning" &&
            classified.confidence < MULTI_SAMPLE_CONFIDENCE_THRESHOLD &&
            proj.multiSample === true)
          ? MULTI_SAMPLE_DEFAULT_N
          : 0;

    let ctxResult;
    let dataResult;
    if (multiN >= 2) {
      const ms = await runMultiSample(payload, requestedModel, callNimNonStream, {
        n: multiN,
        strategy: "consensus",
        callOpts: { ...callOpts, noCache: true },
      });
      ctxResult = ms.winner.ctx;
      dataResult = ms.winner.data;
      const succeeded = ms.samples.filter((s) => s.data).length;
      console.log(
        `[nim] multi-sample: ${succeeded}/${ms.samples.length} ok in ${ms.totalDurationMs}ms, won by sample #${ms.winner.sampleIndex} (${ms.strategy})`,
      );
    } else {
      const r = await callNimNonStream(payload, requestedModel, callOpts);
      ctxResult = r.ctx;
      dataResult = r.data;
    }
    const cacheTag = ctxResult.cacheLayer ? ` [${ctxResult.cacheLayer}-cache]` : "";
    console.log(`[nim] reply via ${ctxResult.modelName} (key ${ctxResult.keyId})${cacheTag}`);

    // Store in conversation cache
    const responseContent = dataResult.choices?.[0]?.message?.content ?? "";
    await appendToConversationCache(
      { model: requestedModel, messages: body.messages },
      responseContent,
      ctxResult.modelName,
      categories[0] ?? null,
      dataResult.usage?.completion_tokens ?? 0,
    );

    // 6. Optional verifier loop: if the assistant requested file edits and
    //    the operator opted in, run typecheck/lint over the touched files
    //    and record diagnostics for the dashboard. Diagnostics are NOT
    //    automatically fed back into the conversation here — that requires
    //    client-side cooperation since the proxy can't apply edits.
    const toolCalls = dataResult.choices?.[0]?.message?.tool_calls ?? [];
    if (VERIFIER_ON_EDIT && toolCalls.length > 0) {
      const touched = extractTouchedFiles(toolCalls);
      if (touched.length > 0) {
        const cwd = process.env["NIM_PROJECT_DIR"] || process.cwd();
        // Fire-and-forget; verifier output goes to dashboard, not response.
        runVerifier({ cwd, touchedFiles: touched })
          .then((result) => {
            recordVerifierRun(result, cwd);
            if (result.diagnostics.length > 0) {
              console.log(
                `[nim] verifier: ${formatDiagnosticsForModel(result).split("\n")[0]}`,
              );
            }
          })
          .catch((e) => {
            console.warn(`[nim] verifier failed: ${e instanceof Error ? e.message : String(e)}`);
          });
      }
    }

    res.json(openAIToAnthropic(dataResult, requestedModel));
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
