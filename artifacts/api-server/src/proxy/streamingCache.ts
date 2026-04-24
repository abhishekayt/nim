/**
 * Streaming response cache (buffer-and-replay).
 *
 * The existing `cache.ts` only caches non-streaming responses. Streaming
 * requests bypass the cache entirely, which means agentic clients that
 * stream the same expensive prompt twice pay for it both times.
 *
 * Strategy:
 *   - On a stream MISS, we proxy the upstream stream chunk-by-chunk to
 *     the client *and* buffer chunks (with relative timestamps) into an
 *     entry. When the stream closes, we commit the entry to the cache.
 *   - On a stream HIT, we replay the buffered chunks, optionally
 *     compressing the inter-chunk delays so the client doesn't re-pay
 *     the original wall-clock latency.
 *
 * Tool-bearing requests are NOT cached for the same reason as in the
 * non-stream cache (agentic state changes turn-to-turn).
 */

import { createHash } from "node:crypto";
import type { OpenAIRequest } from "./translator";

interface BufferedChunk {
  /** Raw chunk bytes as written to the client. */
  data: Uint8Array;
  /** Milliseconds since the entry started. */
  delayMs: number;
}

interface StreamEntry {
  chunks: BufferedChunk[];
  modelName: string;
  finishedAt: number;
  expiresAt: number;
}

function positiveIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

const TTL_MS = positiveIntEnv("NIM_STREAM_CACHE_TTL_MS", 60_000);
const MAX_ENTRIES = positiveIntEnv("NIM_STREAM_CACHE_MAX", 100);
const REPLAY_DELAY_FACTOR = (() => {
  const raw = process.env["NIM_STREAM_CACHE_REPLAY_FACTOR"];
  const n = raw ? Number(raw) : 0.05; // 5% of original delay by default
  return Number.isFinite(n) && n >= 0 && n <= 1 ? n : 0.05;
})();
const REPLAY_DELAY_CAP_MS = positiveIntEnv("NIM_STREAM_CACHE_REPLAY_CAP_MS", 25);
const MAX_BYTES_PER_ENTRY = positiveIntEnv(
  "NIM_STREAM_CACHE_MAX_BYTES",
  2 * 1024 * 1024,
);

const store = new Map<string, StreamEntry>();
let hits = 0;
let misses = 0;
let aborts = 0;

export function streamingCacheKey(
  payload: OpenAIRequest,
  categories: readonly string[] = [],
): string {
  const stable = {
    messages: payload.messages,
    tools: payload.tools,
    tool_choice: payload.tool_choice,
    max_tokens: payload.max_tokens,
    temperature: payload.temperature,
    top_p: payload.top_p,
    stop: payload.stop,
    categories,
    stream: true,
  };
  return createHash("sha256")
    .update(JSON.stringify(stable))
    .digest("hex")
    .slice(0, 32);
}

export function streamingCacheGet(
  key: string,
): { entry: StreamEntry; modelName: string } | null {
  const e = store.get(key);
  if (!e) return null;
  if (e.expiresAt < Date.now()) {
    store.delete(key);
    return null;
  }
  // LRU bump
  store.delete(key);
  store.set(key, e);
  return { entry: e, modelName: e.modelName };
}

/**
 * Replay a cached streaming response into a fresh ReadableStream.
 * Inter-chunk delays are compressed by REPLAY_DELAY_FACTOR (capped).
 */
export function replayStreamingEntry(entry: StreamEntry): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      hits++;
      let prevDelay = 0;
      for (const chunk of entry.chunks) {
        const wait = Math.min(
          REPLAY_DELAY_CAP_MS,
          Math.max(0, (chunk.delayMs - prevDelay) * REPLAY_DELAY_FACTOR),
        );
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        controller.enqueue(chunk.data);
        prevDelay = chunk.delayMs;
      }
      controller.close();
    },
  });
}

interface RecorderHandle {
  /**
   * Tee the upstream stream: forward chunks to the client AND buffer them.
   * Returns the stream the client should consume; commits to the cache
   * when the upstream stream closes successfully.
   */
  wrap(upstream: ReadableStream<Uint8Array>, modelName: string): ReadableStream<Uint8Array>;
}

/**
 * Build a recorder for the given cache key. Caller must `wrap()` the
 * upstream stream and pass the returned stream to the client.
 */
export function recordStreamingResponse(key: string): RecorderHandle {
  return {
    wrap(upstream, modelName) {
      misses++;
      const startedAt = Date.now();
      const buffered: BufferedChunk[] = [];
      let totalBytes = 0;
      let abortBuffering = false;
      let finishedOk = false;

      const reader = upstream.getReader();

      const out = new ReadableStream<Uint8Array>({
        async pull(controller) {
          try {
            const { done, value } = await reader.read();
            if (done) {
              controller.close();
              if (finishedOk || (!abortBuffering && totalBytes > 0)) {
                commitEntry(key, modelName, buffered);
              } else if (abortBuffering) {
                aborts++;
              }
              return;
            }
            if (value) {
              controller.enqueue(value);
              if (!abortBuffering) {
                totalBytes += value.byteLength;
                if (totalBytes > MAX_BYTES_PER_ENTRY) {
                  abortBuffering = true;
                  buffered.length = 0;
                  aborts++;
                } else {
                  // Detect SSE [DONE] sentinel; treat as a clean finish even
                  // if the upstream connection lingers.
                  buffered.push({
                    data: value,
                    delayMs: Date.now() - startedAt,
                  });
                  const tail = bytesToString(value, 256);
                  if (tail.includes("data: [DONE]")) finishedOk = true;
                }
              }
            }
          } catch (err) {
            // Upstream errored mid-stream — don't cache, propagate.
            abortBuffering = true;
            aborts++;
            controller.error(err);
          }
        },
        cancel(reason) {
          // Client went away before stream finished. Don't cache a
          // partial response — it would replay as a truncated answer.
          abortBuffering = true;
          aborts++;
          try {
            reader.cancel(reason);
          } catch {
            /* ignore */
          }
        },
      });

      return out;
    },
  };
}

function commitEntry(
  key: string,
  modelName: string,
  chunks: BufferedChunk[],
): void {
  if (chunks.length === 0) return;
  if (store.size >= MAX_ENTRIES) {
    const oldest = store.keys().next().value;
    if (oldest) store.delete(oldest);
  }
  store.set(key, {
    chunks,
    modelName,
    finishedAt: Date.now(),
    expiresAt: Date.now() + TTL_MS,
  });
}

function bytesToString(bytes: Uint8Array, maxLen: number): string {
  const slice = bytes.byteLength <= maxLen ? bytes : bytes.subarray(bytes.byteLength - maxLen);
  try {
    return new TextDecoder("utf-8", { fatal: false }).decode(slice);
  } catch {
    return "";
  }
}

export function streamingCacheClear(): void {
  store.clear();
}

export function streamingCacheStats(): {
  size: number;
  ttlMs: number;
  hits: number;
  misses: number;
  aborts: number;
  hitRate: number;
} {
  const total = hits + misses;
  return {
    size: store.size,
    ttlMs: TTL_MS,
    hits,
    misses,
    aborts,
    hitRate: total === 0 ? 0 : hits / total,
  };
}
