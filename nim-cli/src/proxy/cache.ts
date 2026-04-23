import { createHash } from "node:crypto";
import type { OpenAINonStreamResponse, OpenAIRequest } from "./translator";

const TTL_MS = Number(process.env["NIM_CACHE_TTL_MS"] ?? 60_000);
const MAX_ENTRIES = Number(process.env["NIM_CACHE_MAX"] ?? 200);

interface Entry {
  data: OpenAINonStreamResponse;
  expiresAt: number;
  modelName: string;
}

const store = new Map<string, Entry>();

export function cacheKey(payload: OpenAIRequest, categories: readonly string[] = []): string {
  // Include tools, messages, model class, and category preference. Don't
  // include the resolved model name — that way cached responses can serve
  // even if rotation picked a different model upstream.
  const stable = {
    messages: payload.messages,
    tools: payload.tools,
    tool_choice: payload.tool_choice,
    max_tokens: payload.max_tokens,
    temperature: payload.temperature,
    top_p: payload.top_p,
    stop: payload.stop,
    categories,
  };
  return createHash("sha256").update(JSON.stringify(stable)).digest("hex").slice(0, 32);
}

export function cacheGet(key: string): { data: OpenAINonStreamResponse; modelName: string } | null {
  const entry = store.get(key);
  if (!entry) return null;
  if (entry.expiresAt < Date.now()) {
    store.delete(key);
    return null;
  }
  // LRU bump
  store.delete(key);
  store.set(key, entry);
  return { data: entry.data, modelName: entry.modelName };
}

export function cacheSet(key: string, data: OpenAINonStreamResponse, modelName: string): void {
  if (store.size >= MAX_ENTRIES) {
    const oldestKey = store.keys().next().value;
    if (oldestKey) store.delete(oldestKey);
  }
  store.set(key, { data, modelName, expiresAt: Date.now() + TTL_MS });
}

export function cacheClear(): void { store.clear(); }
export function cacheStats(): { size: number; ttlMs: number } { return { size: store.size, ttlMs: TTL_MS }; }
