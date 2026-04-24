/**
 * Semantic response cache (embedding-based near-match).
 *
 * Sits in front of the exact-match `cache.ts`. On an exact-cache miss, we
 * compute an embedding of the *last user message* (the part of the
 * request that varies) and look for a recent entry whose embedding is
 * within a cosine-similarity threshold AND whose tools-fingerprint is
 * identical (we never serve a cached answer that was computed under a
 * different tool surface).
 *
 * Conservative defaults:
 *   - threshold 0.92 cosine (high; only near-paraphrases hit)
 *   - never serves cached answers for tool-bearing requests (same as
 *     exact cache; agentic state changes)
 *   - capped at 200 entries with LRU eviction
 *   - per-entry TTL 10 minutes
 */

import { createHash } from "node:crypto";
import type { OpenAINonStreamResponse, OpenAIRequest } from "./translator";
import { embedText, cosineSimilarity } from "./embeddings";

interface SemanticEntry {
  embedding: number[];
  toolsFingerprint: string;
  /** Captured query text (for debugging only). */
  queryPreview: string;
  data: OpenAINonStreamResponse;
  modelName: string;
  expiresAt: number;
  lastHitAt: number;
}

function positiveIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function floatEnv(name: string, fallback: number, min: number, max: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

const TTL_MS = positiveIntEnv("NIM_SEMANTIC_CACHE_TTL_MS", 10 * 60_000);
const MAX_ENTRIES = positiveIntEnv("NIM_SEMANTIC_CACHE_MAX", 200);
const SIM_THRESHOLD = floatEnv("NIM_SEMANTIC_CACHE_THRESHOLD", 0.92, 0, 1);

const store = new Map<string, SemanticEntry>();
let hits = 0;
let misses = 0;
let skipped = 0;

/**
 * Stringify a request's tools array into a stable fingerprint. Two
 * requests with different tool surfaces must NEVER share a cached answer.
 */
function toolsFingerprint(payload: OpenAIRequest): string {
  if (!payload.tools || payload.tools.length === 0) return "no-tools";
  const stable = payload.tools.map((t) => ({
    name: t.function.name,
    parameters: t.function.parameters,
  }));
  return createHash("sha256")
    .update(JSON.stringify(stable))
    .digest("hex")
    .slice(0, 16);
}

/**
 * Extract the text portion of the last user message — the natural query
 * to embed.
 */
function extractQueryText(payload: OpenAIRequest): string | null {
  const messages = payload.messages ?? [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (!m || m.role !== "user") continue;
    if (typeof m.content === "string") return m.content;
    if (Array.isArray(m.content)) {
      const parts: string[] = [];
      for (const block of m.content as Array<{ type?: string; text?: string }>) {
        if (block?.type === "text" && typeof block.text === "string") {
          parts.push(block.text);
        }
      }
      if (parts.length > 0) return parts.join("\n");
    }
  }
  return null;
}

export interface SemanticCacheLookupResult {
  data: OpenAINonStreamResponse;
  modelName: string;
  similarity: number;
}

/**
 * Try to resolve a request via semantic similarity. Returns null on:
 *   - tool-bearing requests (we never serve those from cache)
 *   - empty / un-extractable query
 *   - no embedding available (NIM down or no key)
 *   - no entry within the threshold
 */
export async function semanticCacheGet(
  payload: OpenAIRequest,
): Promise<SemanticCacheLookupResult | null> {
  if (payload.tools && payload.tools.length > 0) {
    skipped++;
    return null;
  }

  const queryText = extractQueryText(payload);
  if (!queryText || queryText.trim().length < 8) {
    skipped++;
    return null;
  }

  const fingerprint = toolsFingerprint(payload);
  const queryEmbedding = await embedText(queryText, { input_type: "query" });
  if (!queryEmbedding) {
    skipped++;
    return null;
  }

  const now = Date.now();
  let bestKey: string | null = null;
  let bestSim = -Infinity;
  let bestEntry: SemanticEntry | null = null;

  for (const [key, entry] of store) {
    if (entry.expiresAt < now) {
      store.delete(key);
      continue;
    }
    if (entry.toolsFingerprint !== fingerprint) continue;
    const sim = cosineSimilarity(queryEmbedding, entry.embedding);
    if (!Number.isFinite(sim)) continue;
    if (sim > bestSim) {
      bestSim = sim;
      bestKey = key;
      bestEntry = entry;
    }
  }

  if (!bestEntry || !bestKey || bestSim < SIM_THRESHOLD) {
    misses++;
    return null;
  }

  bestEntry.lastHitAt = now;
  store.delete(bestKey);
  store.set(bestKey, bestEntry);
  hits++;

  return {
    data: bestEntry.data,
    modelName: bestEntry.modelName,
    similarity: bestSim,
  };
}

/**
 * Store a fresh response for future semantic lookups.
 */
export async function semanticCacheSet(
  payload: OpenAIRequest,
  data: OpenAINonStreamResponse,
  modelName: string,
): Promise<void> {
  if (payload.tools && payload.tools.length > 0) return;

  const queryText = extractQueryText(payload);
  if (!queryText || queryText.trim().length < 8) return;

  const queryEmbedding = await embedText(queryText, { input_type: "passage" });
  if (!queryEmbedding) return;

  if (store.size >= MAX_ENTRIES) {
    const oldest = store.keys().next().value;
    if (oldest) store.delete(oldest);
  }

  const key = createHash("sha256")
    .update(JSON.stringify({ q: queryText, m: modelName }))
    .digest("hex")
    .slice(0, 32);

  store.set(key, {
    embedding: queryEmbedding,
    toolsFingerprint: toolsFingerprint(payload),
    queryPreview: queryText.slice(0, 200),
    data,
    modelName,
    expiresAt: Date.now() + TTL_MS,
    lastHitAt: 0,
  });
}

export function semanticCacheClear(): void {
  store.clear();
}

export function semanticCacheStats(): {
  size: number;
  ttlMs: number;
  threshold: number;
  hits: number;
  misses: number;
  skipped: number;
  hitRate: number;
} {
  const denom = hits + misses;
  return {
    size: store.size,
    ttlMs: TTL_MS,
    threshold: SIM_THRESHOLD,
    hits,
    misses,
    skipped,
    hitRate: denom === 0 ? 0 : hits / denom,
  };
}
