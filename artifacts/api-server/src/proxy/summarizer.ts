/**
 * Conversation compression.
 *
 * When a conversation crosses the model's context budget we have to drop
 * old turns. The naive strategy ("keep last N tokens, slap a one-line
 * summary on the front") destroys agent performance because:
 *   - It splits a tool_use from its tool_result, leaving the model
 *     looking at orphaned tool calls.
 *   - It truncates fenced code mid-block, so later turns cite invalid
 *     snippets.
 *   - It keeps redundant repeated reads of the same file (Claude Code
 *     re-reads files often) instead of deduping.
 *
 * This implementation:
 *   1. Walks pairs (tool_use ↔ matching tool_result) atomically — they
 *      drop together or stay together.
 *   2. Detects repeated `Read` of the same file path and replaces all
 *      but the most recent with a one-line marker.
 *   3. Preserves fenced code blocks: when truncating a text block, never
 *      break inside a ``` fence.
 *   4. Keeps the last K turns verbatim and produces an extractive
 *      summary of dropped middle turns (no extra LLM call needed).
 */

import type { AnthropicMessage, AnthropicBlock } from "./translator";

const TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4;
const KEEP_LAST_TURNS = 8;

interface SummarizationResult {
  messages: AnthropicMessage[];
  wasSummarized: boolean;
  originalCount: number;
  summaryText: string | null;
  /** Diagnostic info for the dashboard. */
  details: {
    droppedCount: number;
    dedupedReads: number;
    keptCount: number;
  };
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / TOKEN_ESTIMATE_CHARS_PER_TOKEN);
}

function blocksOfMessage(msg: AnthropicMessage): AnthropicBlock[] {
  if (typeof msg.content === "string") {
    return [{ type: "text", text: msg.content }];
  }
  return msg.content;
}

function blockToText(b: AnthropicBlock): string {
  if (b.type === "text") return b.text;
  if (b.type === "thinking") return b.thinking;
  if (b.type === "tool_use") return JSON.stringify(b.input);
  if (b.type === "tool_result") {
    if (typeof b.content === "string") return b.content;
    if (Array.isArray(b.content)) return b.content.map((c) => c.text).join("\n");
    return "";
  }
  if (b.type === "image") return "[image]";
  return "";
}

function estimateMessageTokens(msg: AnthropicMessage): number {
  return blocksOfMessage(msg)
    .map((b) => estimateTokens(blockToText(b)))
    .reduce((a, b) => a + b, 0);
}

/**
 * Truncate text safely: never break inside a fenced code block. If a
 * truncation would land inside a fence, extend to the closing fence (or
 * back off to before the opening fence, whichever is shorter).
 */
function truncateRespectingFences(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;

  // Find positions of all ``` fences.
  const fences: number[] = [];
  const fenceRe = /```/g;
  let m: RegExpExecArray | null;
  while ((m = fenceRe.exec(text)) !== null) fences.push(m.index);

  // Determine if `maxChars` lands inside an open fence (odd # of fences before it).
  const fencesBefore = fences.filter((f) => f < maxChars).length;
  if (fencesBefore % 2 === 0) {
    // Outside a fence; safe to cut.
    return text.slice(0, maxChars) + "\n[…truncated…]";
  }

  // Inside a fence. Find the next closing fence after maxChars.
  const closing = fences.find((f) => f >= maxChars);
  if (closing !== undefined && closing - maxChars < maxChars * 0.3) {
    // Extend the cut to include the closing fence.
    return text.slice(0, closing + 3) + "\n[…truncated…]";
  }

  // Otherwise back up to before the opening fence.
  const opening = [...fences].reverse().find((f) => f < maxChars);
  if (opening !== undefined) {
    return text.slice(0, opening) + "\n[…code block truncated…]";
  }

  return text.slice(0, maxChars) + "\n[…truncated…]";
}

/**
 * Find pairs of (tool_use, tool_result) anchored by tool_use_id.
 * Returns a parallel array marking each message's "pair group". Messages
 * that aren't part of a tool pair get unique groups so they're treated
 * atomically as themselves.
 */
function buildToolPairGroups(messages: AnthropicMessage[]): number[] {
  const groups = new Array<number>(messages.length).fill(-1);
  const idToGroup = new Map<string, number>();
  let nextGroup = 0;

  // First pass: assign tool_use messages.
  for (let i = 0; i < messages.length; i++) {
    const blocks = blocksOfMessage(messages[i]!);
    const useIds = blocks.filter((b) => b.type === "tool_use").map((b) => (b as { id: string }).id);
    if (useIds.length > 0) {
      const g = nextGroup++;
      groups[i] = g;
      for (const id of useIds) idToGroup.set(id, g);
    }
  }

  // Second pass: assign tool_result messages to the same group as their use.
  for (let i = 0; i < messages.length; i++) {
    if (groups[i] !== -1) continue;
    const blocks = blocksOfMessage(messages[i]!);
    const resIds = blocks
      .filter((b) => b.type === "tool_result")
      .map((b) => (b as { tool_use_id: string }).tool_use_id);
    if (resIds.length > 0) {
      const g = idToGroup.get(resIds[0]!);
      if (g !== undefined) {
        groups[i] = g;
        continue;
      }
    }
    groups[i] = nextGroup++;
  }

  return groups;
}

/**
 * Detect repeated `Read` tool_use calls of the same file and mark
 * earlier ones as redundant (we'll replace their tool_result with a
 * compact marker pointing at the latest version).
 */
function findRedundantReads(messages: AnthropicMessage[]): Set<string> {
  // Map filePath -> array of tool_use_ids in order.
  const byFile = new Map<string, string[]>();
  for (const msg of messages) {
    for (const b of blocksOfMessage(msg)) {
      if (b.type !== "tool_use") continue;
      if (!/^read$/i.test(b.name) && !/^read[_-]?file$/i.test(b.name)) continue;
      const input = b.input as { file_path?: unknown; path?: unknown };
      const file = (typeof input.file_path === "string" && input.file_path)
        || (typeof input.path === "string" && input.path);
      if (!file) continue;
      const arr = byFile.get(file) ?? [];
      arr.push(b.id);
      byFile.set(file, arr);
    }
  }

  // For each file with >1 read, mark all but the LAST as redundant.
  const redundant = new Set<string>();
  for (const ids of byFile.values()) {
    if (ids.length <= 1) continue;
    for (let i = 0; i < ids.length - 1; i++) redundant.add(ids[i]!);
  }
  return redundant;
}

/**
 * Apply read-deduplication in place: replace tool_result content for
 * redundant reads with a marker. Returns the count of dedupes applied.
 */
function applyReadDedupe(messages: AnthropicMessage[], redundant: Set<string>): number {
  if (redundant.size === 0) return 0;
  let count = 0;
  for (const msg of messages) {
    const blocks = blocksOfMessage(msg);
    let changed = false;
    const newBlocks: AnthropicBlock[] = blocks.map((b) => {
      if (b.type === "tool_result" && redundant.has(b.tool_use_id)) {
        changed = true;
        count++;
        return {
          type: "tool_result",
          tool_use_id: b.tool_use_id,
          content: "[older read of same file; superseded by a more recent read in this conversation]",
        } as AnthropicBlock;
      }
      return b;
    });
    if (changed) {
      msg.content = newBlocks;
    }
  }
  return count;
}

/**
 * Build an extractive summary of the dropped middle turns.
 */
function summarizeDropped(dropped: AnthropicMessage[]): string {
  if (dropped.length === 0) return "";
  const lines: string[] = [];
  lines.push(`Earlier conversation (${dropped.length} message${dropped.length === 1 ? "" : "s"} compressed):`);

  // Track files touched, tools used, and high-level intents.
  const filesTouched = new Set<string>();
  const toolsUsed = new Map<string, number>();
  const userIntents: string[] = [];

  for (const msg of dropped) {
    for (const b of blocksOfMessage(msg)) {
      if (b.type === "tool_use") {
        toolsUsed.set(b.name, (toolsUsed.get(b.name) ?? 0) + 1);
        const input = b.input as { file_path?: unknown; path?: unknown };
        const file = (typeof input.file_path === "string" && input.file_path)
          || (typeof input.path === "string" && input.path);
        if (file) filesTouched.add(file);
      } else if (b.type === "text" && msg.role === "user") {
        const txt = b.text.trim();
        if (txt.length > 0 && txt.length < 400) {
          userIntents.push(truncateRespectingFences(txt, 200));
        } else if (txt.length >= 400) {
          userIntents.push(truncateRespectingFences(txt, 200));
        }
      }
    }
  }

  if (userIntents.length > 0) {
    lines.push("User said earlier:");
    for (const u of userIntents.slice(-5)) {
      lines.push(`  • ${u.replace(/\n+/g, " ")}`);
    }
  }
  if (toolsUsed.size > 0) {
    const toolList = [...toolsUsed.entries()].map(([n, c]) => `${n}×${c}`).join(", ");
    lines.push(`Tools used: ${toolList}`);
  }
  if (filesTouched.size > 0) {
    const files = [...filesTouched].slice(0, 12).join(", ");
    const more = filesTouched.size > 12 ? ` (+${filesTouched.size - 12} more)` : "";
    lines.push(`Files referenced: ${files}${more}`);
  }
  return lines.join("\n");
}

/**
 * Smart conversation compression.
 *
 * @param messages       Full conversation, oldest first.
 * @param contextWindow  Total token capacity of the model.
 * @param thresholdPercent Fraction of capacity at which compression triggers.
 */
export function summarizeConversation(
  messages: AnthropicMessage[],
  contextWindow: number,
  thresholdPercent = 0.8,
): SummarizationResult {
  const threshold = Math.floor(contextWindow * thresholdPercent);
  const totalTokens = messages.reduce((acc, m) => acc + estimateMessageTokens(m), 0);

  if (totalTokens <= threshold) {
    return {
      messages,
      wasSummarized: false,
      originalCount: messages.length,
      summaryText: null,
      details: { droppedCount: 0, dedupedReads: 0, keptCount: messages.length },
    };
  }

  // Step 1: dedupe repeated file reads in-place (cheap, no info loss).
  // Work on a deep clone so we don't mutate the caller's data.
  const cloned: AnthropicMessage[] = JSON.parse(JSON.stringify(messages));
  const redundant = findRedundantReads(cloned);
  const dedupedReads = applyReadDedupe(cloned, redundant);

  const tokensAfterDedup = cloned.reduce((acc, m) => acc + estimateMessageTokens(m), 0);
  if (tokensAfterDedup <= threshold) {
    return {
      messages: cloned,
      wasSummarized: true,
      originalCount: messages.length,
      summaryText: null,
      details: { droppedCount: 0, dedupedReads, keptCount: cloned.length },
    };
  }

  // Step 2: walk back atomically, keeping pairs together until we fit
  // budget/2 (leave room for summary + new turn).
  const pairGroups = buildToolPairGroups(cloned);
  const keepBudget = Math.floor(threshold / 2);

  const keepIdxs = new Set<number>();
  let keepTokens = 0;
  let lastTurnsKept = 0;

  // First, force-keep the last KEEP_LAST_TURNS user/assistant turns
  // regardless of budget — this avoids the model losing immediate context.
  for (let i = cloned.length - 1; i >= 0 && lastTurnsKept < KEEP_LAST_TURNS; i--) {
    keepIdxs.add(i);
    keepTokens += estimateMessageTokens(cloned[i]!);
    lastTurnsKept++;
  }

  // Now greedily extend backwards, but never split a pair group.
  for (let i = cloned.length - 1; i >= 0; i--) {
    if (keepIdxs.has(i)) continue;
    const g = pairGroups[i]!;
    // Find all indices in this group (excluding i itself for accounting).
    const groupIdxs: number[] = [];
    for (let j = 0; j < cloned.length; j++) {
      if (pairGroups[j] === g) groupIdxs.push(j);
    }
    const groupTokens = groupIdxs
      .filter((j) => !keepIdxs.has(j))
      .reduce((acc, j) => acc + estimateMessageTokens(cloned[j]!), 0);
    if (keepTokens + groupTokens > keepBudget) break;
    for (const j of groupIdxs) keepIdxs.add(j);
    keepTokens += groupTokens;
  }

  const keptIndicesSorted = [...keepIdxs].sort((a, b) => a - b);
  const keptMessages = keptIndicesSorted.map((i) => cloned[i]!);
  const droppedMessages = cloned.filter((_, i) => !keepIdxs.has(i));

  const summary = summarizeDropped(droppedMessages);

  // Sanity: if the kept set still starts with a stray `tool_result`
  // (its `tool_use` got dropped because it lived outside the kept range),
  // strip it — leaving an orphaned tool_result confuses every model.
  while (keptMessages.length > 0) {
    const first = keptMessages[0]!;
    const blocks = blocksOfMessage(first);
    const onlyToolResult =
      blocks.length > 0 && blocks.every((b) => b.type === "tool_result");
    if (!onlyToolResult) break;
    keptMessages.shift();
  }

  const summaryMessage: AnthropicMessage = {
    role: "user",
    content: summary,
  };

  return {
    messages: [summaryMessage, ...keptMessages],
    wasSummarized: true,
    originalCount: messages.length,
    summaryText: summary,
    details: {
      droppedCount: droppedMessages.length,
      dedupedReads,
      keptCount: keptMessages.length,
    },
  };
}

/**
 * Plain-text summarizer (used by conversationCache.ts for sliding-window
 * compaction). Kept for backward compat.
 */
export function summarizeTextMessages(
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
): { summary: string; kept: Array<{ role: string; content: string }> } {
  const total = messages.reduce((acc, m) => acc + estimateTokens(m.content), 0);
  if (total <= maxTokens) return { summary: "", kept: messages };

  const keepBudget = Math.floor(maxTokens / 2);
  let keepCount = 0;
  let keepTokens = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = estimateTokens(messages[i]!.content);
    if (keepTokens + t > keepBudget) break;
    keepTokens += t;
    keepCount++;
  }
  const toSummarize = messages.slice(0, messages.length - keepCount);
  const kept = messages.slice(messages.length - keepCount);

  const summary = toSummarize
    .map((m) => `${m.role}: ${truncateRespectingFences(m.content, 200)}`)
    .join("\n");

  return { summary, kept };
}
