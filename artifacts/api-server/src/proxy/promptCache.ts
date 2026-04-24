/**
 * Prompt cache_control translation.
 *
 * Anthropic clients (including Claude Code) annotate their requests with
 *   { ..., cache_control: { type: "ephemeral" } }
 * on `system` blocks, message content blocks, and tool definitions to mark
 * stable prefixes that the model provider should cache. Open-weight NIM
 * models do not support this natively, but we can use the same markers to
 * key our own response cache against the *prefix* of the request rather
 * than the full request — so a follow-up turn that reuses the same long
 * system prompt still hits the cache for the prefix-only computation.
 *
 * This module:
 *   - Walks the Anthropic request and locates every cache_control marker.
 *   - Returns the ordered list of breakpoint positions.
 *   - Computes a stable hash of the request *up to* a chosen breakpoint
 *     (`prefixHash`) so callers can build cache keys that survive small
 *     suffix edits (the user's new turn).
 */

import { createHash } from "node:crypto";

interface CacheControlMarker {
  type: "ephemeral";
}

interface MaybeCacheable {
  cache_control?: CacheControlMarker;
}

/**
 * Loose shape: any Anthropic-style request with optional system/messages/
 * tools. We deliberately omit an index signature so concrete types (like
 * `AnthropicRequest` in translator.ts) satisfy this constraint.
 */
export type RequestLike = {
  model?: string;
  system?: unknown;
  messages?: unknown;
  tools?: unknown;
};

export interface CacheBreakpoint {
  /** Where in the request the breakpoint sits. */
  scope: "system" | "tool" | "message";
  /** Index inside the scope (e.g. tool index, message index). */
  index: number;
  /** For message scope, also the content-block index when content is array. */
  blockIndex?: number;
}

function hasCacheControl(x: unknown): x is MaybeCacheable {
  return (
    typeof x === "object" &&
    x !== null &&
    "cache_control" in x &&
    typeof (x as { cache_control?: unknown }).cache_control === "object" &&
    (x as { cache_control?: { type?: unknown } }).cache_control?.type === "ephemeral"
  );
}

/**
 * Find every cache_control breakpoint in an Anthropic-style request.
 * Returns them in document order: system blocks first, then tools, then
 * messages.
 */
export function extractCacheBreakpoints(req: RequestLike): CacheBreakpoint[] {
  const out: CacheBreakpoint[] = [];

  // system: can be string (no markers possible) or array of text blocks.
  if (Array.isArray(req.system)) {
    for (let i = 0; i < req.system.length; i++) {
      if (hasCacheControl(req.system[i])) {
        out.push({ scope: "system", index: i });
      }
    }
  }

  // tools: array of { name, description, input_schema, cache_control? }
  if (Array.isArray(req.tools)) {
    for (let i = 0; i < req.tools.length; i++) {
      if (hasCacheControl(req.tools[i])) {
        out.push({ scope: "tool", index: i });
      }
    }
  }

  // messages: each message has content = string | array of blocks.
  if (Array.isArray(req.messages)) {
    for (let mi = 0; mi < req.messages.length; mi++) {
      const m = req.messages[mi] as { content?: unknown };
      const content = m?.content;
      if (Array.isArray(content)) {
        for (let bi = 0; bi < content.length; bi++) {
          if (hasCacheControl(content[bi])) {
            out.push({ scope: "message", index: mi, blockIndex: bi });
          }
        }
      }
    }
  }

  return out;
}

/**
 * Build a deterministic hash of everything in the request *up to and
 * including* the given breakpoint. Used as a cache key prefix that survives
 * suffix edits.
 *
 * If no breakpoint is given, hashes the whole request (equivalent to a
 * full cache key).
 */
export function prefixHash(req: RequestLike, upTo: CacheBreakpoint | null): string {
  // We always include model + system fully, since cache_control is
  // semantically "everything before me must be byte-equal to share cache."
  const payload: Record<string, unknown> = {
    model: req.model ?? null,
  };

  if (!upTo) {
    payload["system"] = req.system ?? null;
    payload["tools"] = req.tools ?? null;
    payload["messages"] = req.messages ?? null;
    return createHash("sha256")
      .update(JSON.stringify(payload))
      .digest("hex")
      .slice(0, 32);
  }

  // Slice each section up to the breakpoint.
  if (upTo.scope === "system") {
    payload["system"] = Array.isArray(req.system)
      ? (req.system as unknown[]).slice(0, upTo.index + 1)
      : req.system ?? null;
  } else {
    payload["system"] = req.system ?? null;
  }

  if (upTo.scope === "tool") {
    payload["tools"] = Array.isArray(req.tools)
      ? (req.tools as unknown[]).slice(0, upTo.index + 1)
      : null;
  } else if (upTo.scope === "system") {
    payload["tools"] = null;
  } else {
    payload["tools"] = req.tools ?? null;
  }

  if (upTo.scope === "message" && Array.isArray(req.messages)) {
    const msgs = req.messages as Array<{ role?: string; content?: unknown }>;
    const truncated = msgs.slice(0, upTo.index + 1).map((m, i) => {
      if (i < upTo.index) return m;
      const content = m.content;
      if (Array.isArray(content) && upTo.blockIndex !== undefined) {
        return { ...m, content: content.slice(0, upTo.blockIndex + 1) };
      }
      return m;
    });
    payload["messages"] = truncated;
  } else if (upTo.scope === "message") {
    payload["messages"] = null;
  }

  return createHash("sha256")
    .update(JSON.stringify(payload))
    .digest("hex")
    .slice(0, 32);
}

/**
 * Convenience: pick the *latest* (i.e. closest to the suffix) cache
 * breakpoint, which is the most useful prefix to key against.
 */
export function pickBestBreakpoint(req: RequestLike): CacheBreakpoint | null {
  const all = extractCacheBreakpoints(req);
  if (all.length === 0) return null;
  return all[all.length - 1] ?? null;
}

/**
 * Strip cache_control markers from the request before sending upstream
 * (the OpenAI-compatible NIM API does not understand them). Returns a
 * deep-cloned request so the caller's copy is untouched.
 */
export function stripCacheControl<T extends RequestLike>(req: T): T {
  const cloned = JSON.parse(JSON.stringify(req)) as T;

  if (Array.isArray(cloned.system)) {
    for (const block of cloned.system as MaybeCacheable[]) {
      if (block && typeof block === "object") delete block.cache_control;
    }
  }
  if (Array.isArray(cloned.tools)) {
    for (const tool of cloned.tools as MaybeCacheable[]) {
      if (tool && typeof tool === "object") delete tool.cache_control;
    }
  }
  if (Array.isArray(cloned.messages)) {
    for (const m of cloned.messages as Array<{ content?: unknown }>) {
      if (Array.isArray(m?.content)) {
        for (const block of m.content as MaybeCacheable[]) {
          if (block && typeof block === "object") delete block.cache_control;
        }
      }
    }
  }
  return cloned;
}
