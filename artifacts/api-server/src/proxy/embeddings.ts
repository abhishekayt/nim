/**
 * NIM embedding client.
 *
 * Calls NVIDIA's `nvidia/nv-embedqa-e5-v5` (or a configurable equivalent)
 * to produce 1024-dim embeddings used by the semantic cache and by the
 * code-aware request augmenter for relevance ranking.
 *
 * The client picks any active NIM key from the same store the proxy uses,
 * so users don't need a separate embedding key. Failures are silent: if
 * NIM is misconfigured or returns an error, the embedding call resolves
 * to `null` and callers fall back to non-embedding behavior.
 */

import { loadConfig, resolveBaseUrl } from "./store";

const EMBED_MODEL = process.env["NIM_EMBED_MODEL"] ?? "nvidia/nv-embedqa-e5-v5";
const EMBED_DIM = 1024;
const EMBED_TIMEOUT_MS = 5_000;

/**
 * Cosine similarity for two same-length vectors. Returns NaN if either
 * vector is zero.
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length || a.length === 0) return NaN;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  if (na === 0 || nb === 0) return NaN;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

interface NimEmbeddingResponse {
  data?: Array<{ embedding?: number[] }>;
  error?: { message?: string };
}

/**
 * Get an embedding for a single text. Returns null if NIM is unavailable
 * or the call fails.
 */
export async function embedText(
  text: string,
  options: { input_type?: "query" | "passage" } = {},
): Promise<number[] | null> {
  if (!text || text.trim().length === 0) return null;

  const cfg = await loadConfig();
  const key = cfg.keys.find((k) => k.enabled && !k.rateLimitedUntil);
  if (!key) return null;

  const baseUrl = resolveBaseUrl(cfg, key);
  const url = `${baseUrl.replace(/\/$/, "")}/embeddings`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), EMBED_TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": `Bearer ${key.key}`,
      },
      body: JSON.stringify({
        model: EMBED_MODEL,
        input: text.length > 8000 ? text.slice(0, 8000) : text,
        input_type: options.input_type ?? "query",
        encoding_format: "float",
      }),
      signal: controller.signal,
    });
    if (!res.ok) return null;
    const json = (await res.json()) as NimEmbeddingResponse;
    const vec = json.data?.[0]?.embedding;
    if (!Array.isArray(vec) || vec.length === 0) return null;
    return vec;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Get embeddings for multiple texts in a single request when supported,
 * with a fallback to per-text calls.
 */
export async function embedTexts(
  texts: readonly string[],
  options: { input_type?: "query" | "passage" } = {},
): Promise<Array<number[] | null>> {
  if (texts.length === 0) return [];

  const cfg = await loadConfig();
  const key = cfg.keys.find((k) => k.enabled && !k.rateLimitedUntil);
  if (!key) return texts.map(() => null);

  const baseUrl = resolveBaseUrl(cfg, key);
  const url = `${baseUrl.replace(/\/$/, "")}/embeddings`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), EMBED_TIMEOUT_MS * 2);

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": `Bearer ${key.key}`,
      },
      body: JSON.stringify({
        model: EMBED_MODEL,
        input: texts.map((t) => (t.length > 8000 ? t.slice(0, 8000) : t)),
        input_type: options.input_type ?? "query",
        encoding_format: "float",
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      // Fall back to individual calls
      return Promise.all(texts.map((t) => embedText(t, options)));
    }
    const json = (await res.json()) as NimEmbeddingResponse;
    const out: Array<number[] | null> = [];
    for (let i = 0; i < texts.length; i++) {
      const v = json.data?.[i]?.embedding;
      out.push(Array.isArray(v) && v.length > 0 ? v : null);
    }
    return out;
  } catch {
    return texts.map(() => null);
  } finally {
    clearTimeout(timer);
  }
}

export const EMBEDDING_DIM = EMBED_DIM;
