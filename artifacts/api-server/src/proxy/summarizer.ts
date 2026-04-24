/**
 * Context window summarization.
 *
 * When conversations approach the model's context limit, this module
 * summarizes older messages into a compact system message, preserving
 * the conversation's intent while freeing up token budget.
 */

import type { AnthropicMessage } from "./translator";

interface SummarizationResult {
  messages: AnthropicMessage[];
  wasSummarized: boolean;
  originalCount: number;
  summaryText: string | null;
}

const TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4;

function estimateTokens(text: string): number {
  return Math.ceil(text.length / TOKEN_ESTIMATE_CHARS_PER_TOKEN);
}

function estimateMessageTokens(msg: AnthropicMessage): number {
  if (typeof msg.content === "string") {
    return estimateTokens(msg.content);
  }
  let total = 0;
  for (const block of msg.content) {
    if (block.type === "text") total += estimateTokens(block.text);
    else if (block.type === "thinking") total += estimateTokens(block.thinking);
    else if (block.type === "tool_use") total += estimateTokens(JSON.stringify(block.input));
    else if (block.type === "tool_result") {
      const txt = typeof block.content === "string"
        ? block.content
        : Array.isArray(block.content) ? block.content.map((c) => c.text).join("\n") : "";
      total += estimateTokens(txt);
    }
  }
  return total;
}

/**
 * Summarize a conversation when it exceeds a token threshold.
 * Keeps the most recent messages intact and summarizes older ones.
 */
export function summarizeConversation(
  messages: AnthropicMessage[],
  contextWindow: number,
  thresholdPercent = 0.8,
): SummarizationResult {
  const threshold = Math.floor(contextWindow * thresholdPercent);

  let totalTokens = 0;
  for (const msg of messages) {
    totalTokens += estimateMessageTokens(msg);
  }

  if (totalTokens <= threshold) {
    return {
      messages,
      wasSummarized: false,
      originalCount: messages.length,
      summaryText: null,
    };
  }

  // We need to summarize. Keep the last N messages that fit under threshold/2,
  // and summarize everything before that.
  const keepBudget = Math.floor(threshold / 2);
  let keepCount = 0;
  let keepTokens = 0;

  for (let i = messages.length - 1; i >= 0; i--) {
    const tokens = estimateMessageTokens(messages[i]!);
    if (keepTokens + tokens > keepBudget) break;
    keepTokens += tokens;
    keepCount++;
  }

  const toSummarize = messages.slice(0, messages.length - keepCount);
  const toKeep = messages.slice(messages.length - keepCount);

  // Build a summary of the old messages
  const summaryParts: string[] = [];
  summaryParts.push("Previous conversation summary:");

  for (const msg of toSummarize) {
    const role = msg.role;
    let content = "";
    if (typeof msg.content === "string") {
      content = msg.content;
    } else {
      content = msg.content
        .map((b) => {
          if (b.type === "text") return b.text;
          if (b.type === "thinking") return `[thinking: ${b.thinking.slice(0, 100)}...]`;
          if (b.type === "tool_use") return `[tool: ${b.name}]`;
          if (b.type === "tool_result") {
            const txt = typeof b.content === "string"
              ? b.content
              : Array.isArray(b.content) ? b.content.map((c) => c.text).join("\n") : "";
            return `[result: ${txt.slice(0, 100)}...]`;
          }
          return "";
        })
        .join(" ");
    }

    const truncated = content.length > 200 ? content.slice(0, 200) + "..." : content;
    summaryParts.push(`${role}: ${truncated}`);
  }

  const summaryText = summaryParts.join("\n");
  const summaryMessage: AnthropicMessage = {
    role: "user",
    content: summaryText,
  };

  return {
    messages: [summaryMessage, ...toKeep],
    wasSummarized: true,
    originalCount: messages.length,
    summaryText,
  };
}

/**
 * Summarize a list of text messages (simpler version for conversation cache).
 */
export function summarizeTextMessages(
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
): { summary: string; kept: Array<{ role: string; content: string }> } {
  let totalTokens = 0;
  for (const msg of messages) {
    totalTokens += estimateTokens(msg.content);
  }

  if (totalTokens <= maxTokens) {
    return { summary: "", kept: messages };
  }

  const keepBudget = Math.floor(maxTokens / 2);
  let keepCount = 0;
  let keepTokens = 0;

  for (let i = messages.length - 1; i >= 0; i--) {
    const tokens = estimateTokens(messages[i]!.content);
    if (keepTokens + tokens > keepBudget) break;
    keepTokens += tokens;
    keepCount++;
  }

  const toSummarize = messages.slice(0, messages.length - keepCount);
  const kept = messages.slice(messages.length - keepCount);

  const summary = toSummarize
    .map((m) => `${m.role}: ${m.content.slice(0, 150)}${m.content.length > 150 ? "..." : ""}`)
    .join("\n");

  return { summary, kept };
}
