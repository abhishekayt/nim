import { loadConfig, resolveBaseUrl, type ModelCategory } from "./store";
import {
  pickKey, pickModel,
  markKeyRateLimited, markKeyError, markKeySuccess,
  markModelFailure, markModelSuccess,
} from "./router";
import { withToolCallRetryHint, type OpenAIRequest, type OpenAINonStreamResponse } from "./translator";
import { cacheGet, cacheKey, cacheSet } from "./cache";
import { recordRequest } from "./telemetry";

export interface CallOptions {
  /** Ordered preference of model categories to use for routing. */
  categories?: ModelCategory[];
  /** Disable response caching for this request. */
  noCache?: boolean;
  /** Signals from the classifier (for telemetry). */
  signals?: string[];
}

export interface CallContext {
  keyId: string;
  modelName: string;
  modelId: string;
}

export interface NonStreamResult {
  ctx: CallContext;
  data: OpenAINonStreamResponse;
}

export interface StreamResult {
  ctx: CallContext;
  body: ReadableStream<Uint8Array>;
}

const MAX_KEY_ATTEMPTS = 3;
const MAX_MODEL_ATTEMPTS = 4;

class UpstreamError extends Error {
  constructor(public status: number, public body: string) { super(`Upstream ${status}: ${body.slice(0, 200)}`); }
}

function isRateLimit(status: number, body: string): boolean {
  if (status === 429) return true;
  if (status === 403 && /quota|rate|limit/i.test(body)) return true;
  return false;
}

function isModelError(status: number, body: string): boolean {
  // Treat invalid model / not found / 5xx as model errors worth switching
  if (status === 404) return true;
  if (status >= 500) return true;
  if (status === 400 && /model/i.test(body)) return true;
  return false;
}

async function callOnce(opts: {
  baseUrl: string;
  apiKey: string;
  payload: OpenAIRequest;
  stream: boolean;
}): Promise<{ ok: true; res: Response } | { ok: false; status: number; body: string }> {
  const url = `${opts.baseUrl.replace(/\/$/, "")}/chat/completions`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "authorization": `Bearer ${opts.apiKey}`,
      "accept": opts.stream ? "text/event-stream" : "application/json",
    },
    body: JSON.stringify(opts.payload),
  });
  if (!res.ok) {
    const body = await res.text();
    return { ok: false, status: res.status, body };
  }
  return { ok: true, res };
}

/**
 * Inspect a non-stream response and decide if the model botched its tool
 * calling. Two failure modes:
 *   1. tool_calls are present but their `arguments` are not valid JSON
 *   2. no tool_calls but the assistant content looks like a JSON tool call
 *      printed as prose (common with smaller models)
 */
function isMalformedToolCall(data: OpenAINonStreamResponse, hadTools: boolean): boolean {
  if (!hadTools) return false;
  const choice = data.choices?.[0];
  if (!choice) return false;
  const tcs = choice.message?.tool_calls ?? [];
  for (const tc of tcs) {
    try { JSON.parse(tc.function.arguments || "{}"); }
    catch { return true; }
  }
  if (tcs.length === 0) {
    const txt = (choice.message?.content ?? "").trim();
    if (!txt) return false;
    // Heuristic: looks like a fenced JSON tool-call block or raw {"name": ..., "arguments": ...}
    const looksLikeJsonToolCall =
      /```\s*(json|tool[_ ]?call)?[\s\S]*"(name|tool|function)"\s*:[\s\S]*"(arguments|input|parameters)"/i.test(txt) ||
      /^\s*\{[\s\S]*"(name|tool|function)"\s*:[\s\S]*"(arguments|input|parameters)"[\s\S]*\}\s*$/.test(txt);
    if (looksLikeJsonToolCall) return true;
    // Qwen-family failure mode: wraps the call in <tool_call>…</tool_call>
    // (or <function_call>, <tool>) instead of using the function-calling API.
    const looksLikeXmlToolCall =
      /<\s*(tool[_ ]?call|function[_ ]?call|tool|invoke|use[_ ]?tool)\b[^>]*>/i.test(txt);
    if (looksLikeXmlToolCall) return true;
  }
  return false;
}

export async function callNimNonStream(
  payload: OpenAIRequest,
  requestedAnthropicModel: string,
  opts: CallOptions = {},
): Promise<NonStreamResult> {
  const startedAt = Date.now();
  const triedModels = new Set<string>();
  const triedKeysForModel = new Map<string, Set<string>>();
  let lastError: string = "no attempts made";
  const hadTools = !!(payload.tools && payload.tools.length > 0);

  // ----- Response cache (60s) -----
  // Never cache tool-bearing requests: agentic loops depend on the model
  // reacting to changing state, and serving a stale answer to "what's in
  // src/foo.ts now?" produces silently wrong behaviour.
  const cacheable = !opts.noCache && !hadTools;
  const ck = cacheable ? cacheKey(payload, opts.categories ?? []) : null;
  if (ck) {
    const hit = cacheGet(ck);
    if (hit) {
      recordRequest({
        ts: Date.now(),
        modelName: hit.modelName, modelId: null, keyId: null, providerId: null,
        categories: opts.categories ?? [], signals: opts.signals ?? [],
        latencyMs: Date.now() - startedAt,
        status: "cached", cached: true, streaming: false,
        inputTokens: hit.data.usage?.prompt_tokens ?? null,
        outputTokens: hit.data.usage?.completion_tokens ?? null,
        errorMessage: null,
      });
      return { ctx: { keyId: "cache", modelId: "cache", modelName: hit.modelName }, data: hit.data };
    }
  }

  for (let modelAttempt = 0; modelAttempt < MAX_MODEL_ATTEMPTS; modelAttempt++) {
    const cfg = await loadConfig();
    const model = await pickModel(cfg, requestedAnthropicModel, opts.categories);
    if (!model) throw new Error("No models available");
    if (triedModels.has(model.id)) {
      // exhausted models
      break;
    }
    triedModels.add(model.id);
    if (!triedKeysForModel.has(model.id)) triedKeysForModel.set(model.id, new Set());
    const triedKeys = triedKeysForModel.get(model.id)!;

    for (let keyAttempt = 0; keyAttempt < MAX_KEY_ATTEMPTS; keyAttempt++) {
      const cfg2 = await loadConfig();
      const key = await pickKey(cfg2);
      if (!key) throw new Error("No NIM API keys configured. Add at least one in the dashboard.");
      if (triedKeys.has(key.id)) break;
      triedKeys.add(key.id);

      const baseUrl = resolveBaseUrl(cfg2, key);
      let result = await callOnce({
        baseUrl,
        apiKey: key.key,
        payload: { ...payload, model: model.name, stream: false },
        stream: false,
      });

      if (result.ok) {
        let data = (await result.res.json()) as OpenAINonStreamResponse;

        // One-shot retry if the model emitted a malformed tool call.
        if (isMalformedToolCall(data, hadTools)) {
          console.warn(`[nim] ${model.name} returned malformed tool call — retrying once with corrective hint`);
          const retried = await callOnce({
            baseUrl,
            apiKey: key.key,
            payload: { ...withToolCallRetryHint(payload), model: model.name, stream: false },
            stream: false,
          });
          if (retried.ok) {
            const retryData = (await retried.res.json()) as OpenAINonStreamResponse;
            if (!isMalformedToolCall(retryData, hadTools)) {
              data = retryData;
            }
            // If still malformed, return the retried response anyway — Claude
            // Code will surface the parse failure and the user can intervene.
          }
        }

        await markKeySuccess(key.id);
        await markModelSuccess(model.id);
        if (ck) cacheSet(ck, data, model.name);
        recordRequest({
          ts: Date.now(),
          modelName: model.name, modelId: model.id, keyId: key.id, providerId: key.providerId ?? "nim",
          categories: opts.categories ?? [], signals: opts.signals ?? [],
          latencyMs: Date.now() - startedAt,
          status: "ok", cached: false, streaming: false,
          inputTokens: data.usage?.prompt_tokens ?? null,
          outputTokens: data.usage?.completion_tokens ?? null,
          errorMessage: null,
        });
        return { ctx: { keyId: key.id, modelId: model.id, modelName: model.name }, data };
      }

      lastError = `${result.status} ${result.body.slice(0, 200)}`;
      if (isRateLimit(result.status, result.body)) {
        await markKeyRateLimited(key.id, result.body.slice(0, 200));
        continue; // try another key
      }
      if (isModelError(result.status, result.body)) {
        await markModelFailure(model.id, result.body.slice(0, 200));
        break; // switch model
      }
      await markKeyError(key.id, lastError);
      recordRequest({
        ts: Date.now(),
        modelName: model.name, modelId: model.id, keyId: key.id, providerId: key.providerId ?? "nim",
        categories: opts.categories ?? [], signals: opts.signals ?? [],
        latencyMs: Date.now() - startedAt,
        status: "error", cached: false, streaming: false,
        inputTokens: null, outputTokens: null, errorMessage: lastError,
      });
      throw new UpstreamError(result.status, result.body);
    }
  }
  recordRequest({
    ts: Date.now(),
    modelName: "(none)", modelId: null, keyId: null, providerId: null,
    categories: opts.categories ?? [], signals: opts.signals ?? [],
    latencyMs: Date.now() - startedAt,
    status: "error", cached: false, streaming: false,
    inputTokens: null, outputTokens: null, errorMessage: lastError,
  });
  throw new Error(`All keys/models exhausted. Last error: ${lastError}`);
}

export async function callNimStream(
  payload: OpenAIRequest,
  requestedAnthropicModel: string,
  opts: CallOptions = {},
): Promise<StreamResult> {
  const startedAt = Date.now();
  const triedModels = new Set<string>();
  const triedKeysForModel = new Map<string, Set<string>>();
  let lastError: string = "no attempts made";

  for (let modelAttempt = 0; modelAttempt < MAX_MODEL_ATTEMPTS; modelAttempt++) {
    const cfg = await loadConfig();
    const model = await pickModel(cfg, requestedAnthropicModel, opts.categories);
    if (!model) throw new Error("No models available");
    if (triedModels.has(model.id)) break;
    triedModels.add(model.id);
    if (!triedKeysForModel.has(model.id)) triedKeysForModel.set(model.id, new Set());
    const triedKeys = triedKeysForModel.get(model.id)!;

    for (let keyAttempt = 0; keyAttempt < MAX_KEY_ATTEMPTS; keyAttempt++) {
      const cfg2 = await loadConfig();
      const key = await pickKey(cfg2);
      if (!key) throw new Error("No NIM API keys configured. Add at least one in the dashboard.");
      if (triedKeys.has(key.id)) break;
      triedKeys.add(key.id);

      const result = await callOnce({
        baseUrl: resolveBaseUrl(cfg2, key),
        apiKey: key.key,
        payload: { ...payload, model: model.name, stream: true },
        stream: true,
      });

      if (result.ok && result.res.body) {
        await markKeySuccess(key.id);
        await markModelSuccess(model.id);
        recordRequest({
          ts: Date.now(),
          modelName: model.name, modelId: model.id, keyId: key.id, providerId: key.providerId ?? "nim",
          categories: opts.categories ?? [], signals: opts.signals ?? [],
          latencyMs: Date.now() - startedAt,
          status: "ok", cached: false, streaming: true,
          inputTokens: null, outputTokens: null, errorMessage: null,
        });
        return {
          ctx: { keyId: key.id, modelId: model.id, modelName: model.name },
          body: result.res.body,
        };
      }

      if (!result.ok) {
        lastError = `${result.status} ${result.body.slice(0, 200)}`;
        if (isRateLimit(result.status, result.body)) {
          await markKeyRateLimited(key.id, result.body.slice(0, 200));
          continue;
        }
        if (isModelError(result.status, result.body)) {
          await markModelFailure(model.id, result.body.slice(0, 200));
          break;
        }
        await markKeyError(key.id, lastError);
        recordRequest({
          ts: Date.now(),
          modelName: model.name, modelId: model.id, keyId: key.id, providerId: key.providerId ?? "nim",
          categories: opts.categories ?? [], signals: opts.signals ?? [],
          latencyMs: Date.now() - startedAt,
          status: "error", cached: false, streaming: true,
          inputTokens: null, outputTokens: null, errorMessage: lastError,
        });
        throw new UpstreamError(result.status, result.body);
      }
    }
  }
  recordRequest({
    ts: Date.now(),
    modelName: "(none)", modelId: null, keyId: null, providerId: null,
    categories: opts.categories ?? [], signals: opts.signals ?? [],
    latencyMs: Date.now() - startedAt,
    status: "error", cached: false, streaming: true,
    inputTokens: null, outputTokens: null, errorMessage: lastError,
  });
  throw new Error(`All keys/models exhausted. Last error: ${lastError}`);
}
