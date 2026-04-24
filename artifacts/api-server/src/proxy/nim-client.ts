import { loadConfig, resolveBaseUrl, type ModelCategory } from "./store";
import {
  pickKey, pickModel,
  markKeyRateLimited, markKeyError, markKeySuccess,
  markModelFailure, markModelSuccess,
} from "./router";
import { withToolCallRetryHint, type OpenAIRequest, type OpenAINonStreamResponse } from "./translator";
import { cacheGet, cacheKey, cacheSet } from "./cache";
import { recordRequestDb } from "./telemetryDb";
import {
  isMalformedToolCallEnhanced,
  extractToolCallsFromText,
  validateToolCalls,
  repairResponseToolCalls,
} from "./toolValidator";
import { semanticCacheGet, semanticCacheSet } from "./semanticCache";
import {
  streamingCacheKey,
  streamingCacheGet,
  replayStreamingEntry,
  recordStreamingResponse,
} from "./streamingCache";
import { type CascadePlan, recordCascadeEvent } from "./cascade";

export interface CallOptions {
  /** Ordered preference of model categories to use for routing. */
  categories?: ModelCategory[];
  /** Disable response caching for this request. */
  noCache?: boolean;
  /** Signals from the classifier (for telemetry). */
  signals?: string[];
  /** Cascade escalation plan; overrides `categories` per attempt. */
  cascadePlan?: CascadePlan;
  /** Disable semantic cache lookup for this request. */
  noSemanticCache?: boolean;
}

export interface CallContext {
  keyId: string;
  modelName: string;
  modelId: string;
  /** Set if the response was served from a cache layer. */
  cacheLayer?: "exact" | "semantic" | "stream";
  /** Cosine similarity for semantic-cache hits. */
  semanticSimilarity?: number;
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

function isMalformedToolCall(data: OpenAINonStreamResponse, hadTools: boolean): boolean {
  return isMalformedToolCallEnhanced(data, hadTools);
}

/**
 * Resolve which categories to ask the router for at a given attempt
 * within the cascade plan. Falls back to the caller-provided categories.
 */
function categoriesForAttempt(
  opts: CallOptions,
  attempt: number,
): ModelCategory[] | undefined {
  if (opts.cascadePlan && opts.cascadePlan.tiers.length > 0) {
    const tier = opts.cascadePlan.tiers[
      Math.min(attempt, opts.cascadePlan.tiers.length - 1)
    ]!;
    return [tier];
  }
  return opts.categories;
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

  // ----- Layer 1: exact response cache (60s) -----
  // Never cache tool-bearing requests: agentic loops depend on the model
  // reacting to changing state.
  const cacheable = !opts.noCache && !hadTools;
  const ck = cacheable ? cacheKey(payload, opts.categories ?? []) : null;
  if (ck) {
    const hit = cacheGet(ck);
    if (hit) {
      await recordRequestDb({
        ts: Date.now(),
        modelName: hit.modelName, modelId: null, keyId: null, providerId: null,
        categories: opts.categories ?? [], signals: opts.signals ?? [],
        latencyMs: Date.now() - startedAt,
        status: "cached", cached: true, streaming: false,
        inputTokens: hit.data.usage?.prompt_tokens ?? null,
        outputTokens: hit.data.usage?.completion_tokens ?? null,
        errorMessage: null,
      });
      return {
        ctx: { keyId: "cache", modelId: "cache", modelName: hit.modelName, cacheLayer: "exact" },
        data: hit.data,
      };
    }
  }

  // ----- Layer 2: semantic cache -----
  if (cacheable && !opts.noSemanticCache) {
    const semHit = await semanticCacheGet(payload);
    if (semHit) {
      await recordRequestDb({
        ts: Date.now(),
        modelName: semHit.modelName, modelId: null, keyId: null, providerId: null,
        categories: opts.categories ?? [], signals: [...(opts.signals ?? []), `sem-cache:${semHit.similarity.toFixed(3)}`],
        latencyMs: Date.now() - startedAt,
        status: "cached-semantic", cached: true, streaming: false,
        inputTokens: semHit.data.usage?.prompt_tokens ?? null,
        outputTokens: semHit.data.usage?.completion_tokens ?? null,
        errorMessage: null,
      });
      return {
        ctx: {
          keyId: "semantic-cache",
          modelId: "semantic-cache",
          modelName: semHit.modelName,
          cacheLayer: "semantic",
          semanticSimilarity: semHit.similarity,
        },
        data: semHit.data,
      };
    }
  }

  for (let modelAttempt = 0; modelAttempt < MAX_MODEL_ATTEMPTS; modelAttempt++) {
    const cfg = await loadConfig();
    const cats = categoriesForAttempt(opts, modelAttempt);
    const model = await pickModel(cfg, requestedAnthropicModel, cats);
    if (!model) throw new Error("No models available");
    if (triedModels.has(model.id)) {
      // exhausted models within this tier; cascade plan will give us a
      // different tier on the next attempt iteration if available.
      if (opts.cascadePlan && modelAttempt < opts.cascadePlan.tiers.length - 1) continue;
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
        let toolRetries = 0;
        let repairsApplied = 0;

        const toolSchemas = payload.tools?.map((t) => ({
          name: t.function.name,
          description: t.function.description,
          parameters: t.function.parameters,
        })) ?? [];

        // Pass 1: local repair (free, no upstream call).
        if (hadTools && data.choices?.[0]?.message?.tool_calls?.length) {
          const repaired = repairResponseToolCalls(data, toolSchemas);
          if (repaired.repaired) {
            data = repaired.data;
            repairsApplied++;
            console.warn(`[nim] ${model.name} tool calls auto-repaired: ${repaired.notes.join("; ")}`);
          }
        }

        // Pass 2: malformed-call detection → upstream retry with hint.
        if (isMalformedToolCall(data, hadTools)) {
          console.warn(`[nim] ${model.name} returned malformed tool call — retrying with corrective hint`);
          toolRetries++;
          const retried = await callOnce({
            baseUrl,
            apiKey: key.key,
            payload: { ...withToolCallRetryHint(payload), model: model.name, stream: false },
            stream: false,
          });
          if (retried.ok) {
            let retryData = (await retried.res.json()) as OpenAINonStreamResponse;
            // Try local repair on the retry too.
            if (retryData.choices?.[0]?.message?.tool_calls?.length) {
              const r2 = repairResponseToolCalls(retryData, toolSchemas);
              if (r2.repaired) {
                retryData = r2.data;
                repairsApplied++;
              }
            }
            if (!isMalformedToolCall(retryData, hadTools)) {
              data = retryData;
            } else {
              // Last resort: extract tool calls from prose content.
              const extracted = extractToolCallsFromText(retryData);
              if (extracted) {
                data = extracted;
                console.warn(`[nim] extracted tool calls from text for ${model.name}`);
              }
            }
          }
        }

        // Pass 3: full schema validation; treat failure as a soft warning
        // (we already retried once; further retries hurt latency too much).
        if (hadTools && data.choices?.[0]?.message?.tool_calls?.length) {
          const validation = validateToolCalls(data, toolSchemas);
          if (!validation.valid) {
            console.warn(`[nim] tool validation failed for ${model.name}: ${validation.errors.join(", ")}`);
            // If we have cascade headroom, treat this as a model failure
            // and escalate to the next tier.
            if (
              opts.cascadePlan &&
              modelAttempt < opts.cascadePlan.tiers.length - 1
            ) {
              await markModelFailure(model.id, `tool validation: ${validation.errors[0] ?? "unknown"}`);
              break; // try next tier
            }
          }
        }

        await markKeySuccess(key.id);
        await markModelSuccess(model.id);
        if (ck) cacheSet(ck, data, model.name);
        // Best-effort populate the semantic cache for future near-matches.
        if (cacheable) {
          void semanticCacheSet(payload, data, model.name).catch(() => undefined);
        }
        if (opts.cascadePlan) {
          recordCascadeEvent({
            plan: opts.cascadePlan,
            winningTier: modelAttempt,
            attempts: modelAttempt + 1,
          });
        }
        await recordRequestDb({
          ts: Date.now(),
          modelName: model.name, modelId: model.id, keyId: key.id, providerId: key.providerId ?? "nim",
          categories: opts.categories ?? [], signals: opts.signals ?? [],
          latencyMs: Date.now() - startedAt,
          status: "ok", cached: false, streaming: false,
          inputTokens: data.usage?.prompt_tokens ?? null,
          outputTokens: data.usage?.completion_tokens ?? null,
          errorMessage: null,
          toolRetries,
          repairsApplied,
        });
        return { ctx: { keyId: key.id, modelId: model.id, modelName: model.name }, data };
      }

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
      await recordRequestDb({
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
  if (opts.cascadePlan) {
    recordCascadeEvent({
      plan: opts.cascadePlan,
      winningTier: -1,
      attempts: triedModels.size,
    });
  }
  await recordRequestDb({
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
  const hadTools = !!(payload.tools && payload.tools.length > 0);

  // ----- Streaming cache (only when no tools, no opt-out) -----
  const streamCacheable = !opts.noCache && !hadTools;
  const sck = streamCacheable ? streamingCacheKey(payload, opts.categories ?? []) : null;
  if (sck) {
    const hit = streamingCacheGet(sck);
    if (hit) {
      await recordRequestDb({
        ts: Date.now(),
        modelName: hit.modelName, modelId: null, keyId: null, providerId: null,
        categories: opts.categories ?? [], signals: [...(opts.signals ?? []), "stream-cache"],
        latencyMs: Date.now() - startedAt,
        status: "cached", cached: true, streaming: true,
        inputTokens: null, outputTokens: null, errorMessage: null,
      });
      return {
        ctx: {
          keyId: "stream-cache",
          modelId: "stream-cache",
          modelName: hit.modelName,
          cacheLayer: "stream",
        },
        body: replayStreamingEntry(hit.entry),
      };
    }
  }

  for (let modelAttempt = 0; modelAttempt < MAX_MODEL_ATTEMPTS; modelAttempt++) {
    const cfg = await loadConfig();
    const cats = categoriesForAttempt(opts, modelAttempt);
    const model = await pickModel(cfg, requestedAnthropicModel, cats);
    if (!model) throw new Error("No models available");
    if (triedModels.has(model.id)) {
      if (opts.cascadePlan && modelAttempt < opts.cascadePlan.tiers.length - 1) continue;
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

      const result = await callOnce({
        baseUrl: resolveBaseUrl(cfg2, key),
        apiKey: key.key,
        payload: { ...payload, model: model.name, stream: true },
        stream: true,
      });

      if (result.ok && result.res.body) {
        await markKeySuccess(key.id);
        await markModelSuccess(model.id);
        if (opts.cascadePlan) {
          recordCascadeEvent({
            plan: opts.cascadePlan,
            winningTier: modelAttempt,
            attempts: modelAttempt + 1,
          });
        }
        await recordRequestDb({
          ts: Date.now(),
          modelName: model.name, modelId: model.id, keyId: key.id, providerId: key.providerId ?? "nim",
          categories: opts.categories ?? [], signals: opts.signals ?? [],
          latencyMs: Date.now() - startedAt,
          status: "ok", cached: false, streaming: true,
          inputTokens: null, outputTokens: null, errorMessage: null,
        });
        // Wrap with the streaming-cache recorder when eligible.
        const finalBody = sck
          ? recordStreamingResponse(sck).wrap(result.res.body, model.name)
          : result.res.body;
        return {
          ctx: { keyId: key.id, modelId: model.id, modelName: model.name },
          body: finalBody,
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
        await recordRequestDb({
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
  if (opts.cascadePlan) {
    recordCascadeEvent({
      plan: opts.cascadePlan,
      winningTier: -1,
      attempts: triedModels.size,
    });
  }
  await recordRequestDb({
    ts: Date.now(),
    modelName: "(none)", modelId: null, keyId: null, providerId: null,
    categories: opts.categories ?? [], signals: opts.signals ?? [],
    latencyMs: Date.now() - startedAt,
    status: "error", cached: false, streaming: true,
    inputTokens: null, outputTokens: null, errorMessage: lastError,
  });
  throw new Error(`All keys/models exhausted. Last error: ${lastError}`);
}
