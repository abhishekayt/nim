import { splitThinking } from "./thinking";
import { buildSystemPrompt, getRecommendedMaxTokens, getContextWindow } from "./systemPrompts";

type Json = unknown;

export interface AnthropicTextBlock { type: "text"; text: string }
export interface AnthropicThinkingBlock { type: "thinking"; thinking: string; signature?: string }
export interface AnthropicImageBlock { type: "image"; source: { type: "base64"; media_type: string; data: string } | { type: "url"; url: string } }
export interface AnthropicToolUseBlock { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
export interface AnthropicToolResultBlock { type: "tool_result"; tool_use_id: string; content?: string | Array<{ type: "text"; text: string }>; is_error?: boolean }
export type AnthropicBlock = AnthropicTextBlock | AnthropicThinkingBlock | AnthropicImageBlock | AnthropicToolUseBlock | AnthropicToolResultBlock;

export interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | AnthropicBlock[];
}

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

export interface AnthropicRequest {
  model: string;
  messages: AnthropicMessage[];
  system?: string | Array<{ type: "text"; text: string }>;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  stream?: boolean;
  tools?: AnthropicTool[];
  tool_choice?: { type: "auto" | "any" | "tool"; name?: string; disable_parallel_tool_use?: boolean };
  metadata?: Record<string, unknown>;
}

/**
 * Default max_tokens cap when the client does not specify one. Several NIM
 * models silently default to 1024 tokens, which truncates real coding
 * responses mid-file and looks like "the model gave up." 8192 matches what
 * Claude Code sends for most requests.
 */
const DEFAULT_MAX_TOKENS = Number(process.env["NIM_DEFAULT_MAX_TOKENS"] ?? 8192);

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | Array<Json> | null;
  /** Reasoning trace from thinking models (Kimi K2.5, DeepSeek-R1, etc.). */
  reasoning_content?: string | null;
  reasoning?: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
  name?: string;
}

export interface OpenAIRequest {
  model: string;
  messages: OpenAIMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  stream?: boolean;
  tools?: Array<{ type: "function"; function: { name: string; description?: string; parameters: Record<string, unknown> } }>;
  tool_choice?: "auto" | "none" | "required" | { type: "function"; function: { name: string } };
  /**
   * Default false when the proxy attaches tools — most agentic clients
   * (including Claude Code) expect strict sequential tool use, and several
   * open models hallucinate parallel calls otherwise.
   */
  parallel_tool_calls?: boolean;
}

function systemToString(sys: AnthropicRequest["system"]): string | null {
  if (!sys) return null;
  if (typeof sys === "string") return sys;
  return sys.map((s) => s.text).join("\n\n");
}

const TOOL_USE_ADDENDUM = `
# CRITICAL TOOL-USE RULES (MUST FOLLOW)

You have access to function-calling tools. When you need to take an action you MUST emit a real tool call via the function-calling API. NEVER:
- Print tool calls as JSON in your prose response (e.g. \`{"name": "Edit", "arguments": ...}\`)
- Wrap tool calls in \`\`\`json fences or markdown
- Describe what you "would" call — actually call it
- Invent tool names or parameters that aren't in the provided schema
- Stop mid-task with "I will now use the X tool" without actually calling it

When calling a tool:
- Pass arguments as a valid JSON object that exactly matches the tool's input schema
- Required parameters must be present; optional ones can be omitted
- For file edits, prefer one focused change at a time over giant multi-file rewrites
- After a tool returns, continue with the next step or finish — do not re-explain what just happened

When responding to the user without a tool, write plainly. Do not narrate your tool plans.`.trim();

const RETRY_FIX_TOOL_CALL = `
# YOUR PREVIOUS RESPONSE WAS INVALID

Your last reply contained a tool call that could not be parsed (invalid JSON arguments) OR you described a tool call as text instead of actually invoking it.

Retry now. This time:
- Use the function-calling API to invoke the tool — do NOT print the call as text
- Emit valid JSON for arguments — no trailing commas, no comments, no \`\`\` fences
- If you cannot make a valid call, respond in plain prose explaining why instead`.trim();

/** Prepend tool-use guidance to the system prompt when tools are present. */
export function augmentSystemForTools(req: AnthropicRequest): AnthropicRequest {
  if (!req.tools || req.tools.length === 0) return req;
  const existing = systemToString(req.system) ?? "";
  const merged = existing
    ? `${existing}\n\n${TOOL_USE_ADDENDUM}`
    : TOOL_USE_ADDENDUM;
  return { ...req, system: merged };
}

/**
 * Build a system prompt with per-model template injection.
 */
export function buildModelSystemPrompt(
  baseSystem: string | null,
  modelName: string,
  hasTools: boolean,
  enableThinking?: boolean,
): string {
  return buildSystemPrompt(baseSystem, modelName, {
    hasTools,
    enableThinking,
    toolUseAddendum: TOOL_USE_ADDENDUM,
  });
}

/**
 * Tool descriptions in MCP / Claude Code can be enormous (4–10 KB each), and
 * very small models drown when 30 of them stack up in the system prompt.
 *
 * BUT: Claude Code's tool descriptions encode critical correctness rules
 * (e.g. the Edit tool's "old_string must be unique" guidance, the Bash
 * tool's "no `cd`" rule). Truncating them silently makes capable models
 * commit the exact mistakes those rules were written to prevent.
 *
 * So this is OFF by default. Opt in with NIM_TRUNCATE_TOOL_DESCRIPTIONS=1
 * if you're routing to a tiny model that can't handle the full prompt.
 */
const TOOL_DESC_MAX = Number(process.env["NIM_TOOL_DESC_MAX"] ?? 600);

export function shortenToolDescription(desc: string | undefined): string | undefined {
  if (!desc) return desc;
  if (process.env["NIM_TRUNCATE_TOOL_DESCRIPTIONS"] !== "1") return desc;
  if (desc.length <= TOOL_DESC_MAX) return desc;
  // Prefer to cut at end-of-sentence boundary near the limit
  const slice = desc.slice(0, TOOL_DESC_MAX);
  const lastBoundary = Math.max(slice.lastIndexOf(". "), slice.lastIndexOf("\n\n"));
  const cut = lastBoundary > TOOL_DESC_MAX * 0.5 ? lastBoundary + 1 : TOOL_DESC_MAX;
  return desc.slice(0, cut).trimEnd() + " […]";
}

/** Add a one-shot corrective system message for the retry attempt. */
export function withToolCallRetryHint(payload: OpenAIRequest): OpenAIRequest {
  const messages = [...payload.messages];
  const sysIdx = messages.findIndex((m) => m.role === "system");
  if (sysIdx >= 0) {
    const sysContent = typeof messages[sysIdx]!.content === "string" ? messages[sysIdx]!.content as string : "";
    messages[sysIdx] = { ...messages[sysIdx]!, content: `${sysContent}\n\n${RETRY_FIX_TOOL_CALL}` };
  } else {
    messages.unshift({ role: "system", content: RETRY_FIX_TOOL_CALL });
  }
  return { ...payload, messages };
}

function blocksToOpenAIContent(blocks: AnthropicBlock[]): { content: string | Array<Json>; toolCalls: OpenAIToolCall[]; toolResults: Array<{ id: string; text: string }> } {
  const parts: Array<Json> = [];
  const toolCalls: OpenAIToolCall[] = [];
  const toolResults: Array<{ id: string; text: string }> = [];
  let textOnly = "";
  let onlyText = true;

  for (const b of blocks) {
    if (b.type === "text") {
      textOnly += (textOnly ? "\n" : "") + b.text;
      parts.push({ type: "text", text: b.text });
    } else if (b.type === "image") {
      onlyText = false;
      const url = b.source.type === "base64" ? `data:${b.source.media_type};base64,${b.source.data}` : b.source.url;
      parts.push({ type: "image_url", image_url: { url } });
    } else if (b.type === "tool_use") {
      onlyText = false;
      toolCalls.push({
        id: b.id,
        type: "function",
        function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) },
      });
    } else if (b.type === "tool_result") {
      onlyText = false;
      const text = typeof b.content === "string"
        ? b.content
        : Array.isArray(b.content)
          ? b.content.map((c) => c.text).join("\n")
          : "";
      toolResults.push({ id: b.tool_use_id, text });
    }
  }

  return {
    content: onlyText ? textOnly : parts,
    toolCalls,
    toolResults,
  };
}

export function anthropicToOpenAI(req: AnthropicRequest, model: string): OpenAIRequest {
  const messages: OpenAIMessage[] = [];
  const sys = systemToString(req.system);
  if (sys) messages.push({ role: "system", content: sys });

  for (const m of req.messages) {
    if (typeof m.content === "string") {
      messages.push({ role: m.role, content: m.content });
      continue;
    }
    const { content, toolCalls, toolResults } = blocksToOpenAIContent(m.content);

    // Emit tool results first (as separate `tool` messages) before any new user/assistant content
    for (const tr of toolResults) {
      messages.push({ role: "tool", tool_call_id: tr.id, content: tr.text });
    }

    if (m.role === "assistant") {
      const msg: OpenAIMessage = { role: "assistant" };
      const hasContent = typeof content === "string" ? content.length > 0 : (content as Array<Json>).length > 0;
      if (hasContent) msg.content = content;
      else msg.content = null;
      if (toolCalls.length > 0) msg.tool_calls = toolCalls;
      if (hasContent || toolCalls.length > 0) messages.push(msg);
    } else {
      // user
      const hasContent = typeof content === "string" ? content.length > 0 : (content as Array<Json>).length > 0;
      if (hasContent) messages.push({ role: "user", content });
    }
  }

  const out: OpenAIRequest = {
    model,
    messages,
    stream: req.stream,
  };
  // Always send a max_tokens. NIM defaults are tiny (often 1024) and
  // truncate real coding answers mid-file.
  out.max_tokens = typeof req.max_tokens === "number" && req.max_tokens > 0
    ? req.max_tokens
    : getRecommendedMaxTokens(model) || DEFAULT_MAX_TOKENS;
  if (typeof req.temperature === "number") out.temperature = req.temperature;
  if (typeof req.top_p === "number") out.top_p = req.top_p;
  if (req.stop_sequences && req.stop_sequences.length > 0) out.stop = req.stop_sequences;

  if (req.tools && req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: shortenToolDescription(t.description), parameters: t.input_schema },
    }));
    if (req.tool_choice) {
      if (req.tool_choice.type === "auto") out.tool_choice = "auto";
      else if (req.tool_choice.type === "any") out.tool_choice = "required";
      else if (req.tool_choice.type === "tool" && req.tool_choice.name)
        out.tool_choice = { type: "function", function: { name: req.tool_choice.name } };
    }
    // Default to sequential tool calls. Claude Code's loop is built around
    // call → result → reason → next call, and several open models hallucinate
    // duplicate parallel calls when left to their own devices. The client can
    // still opt back in with tool_choice.disable_parallel_tool_use === false.
    if (req.tool_choice?.disable_parallel_tool_use === false) {
      out.parallel_tool_calls = true;
    } else {
      out.parallel_tool_calls = false;
    }
  }

  return out;
}

export interface OpenAINonStreamResponse {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: "assistant";
      content: string | null;
      reasoning_content?: string | null;
      reasoning?: string | null;
      tool_calls?: OpenAIToolCall[];
    };
    finish_reason: string | null;
  }>;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

function mapStopReason(r: string | null | undefined): "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" {
  switch (r) {
    case "length": return "max_tokens";
    case "tool_calls": return "tool_use";
    case "stop":
    default: return "end_turn";
  }
}

export function openAIToAnthropic(resp: OpenAINonStreamResponse, requestedModel: string): {
  id: string;
  type: "message";
  role: "assistant";
  model: string;
  content: AnthropicBlock[];
  stop_reason: string;
  stop_sequence: null;
  usage: { input_tokens: number; output_tokens: number };
} {
  const choice = resp.choices[0];
  const blocks: AnthropicBlock[] = [];
  // Reasoning models (Kimi K2.5, DeepSeek-R1, GPT-OSS, Qwen3-thinking, etc.)
  // return their chain-of-thought in a separate `reasoning_content` /
  // `reasoning` field rather than inline <think> tags. Surface it as a
  // proper Anthropic thinking block so Claude Code can display it.
  const reasoning =
    (typeof choice?.message.reasoning_content === "string" ? choice.message.reasoning_content : "") ||
    (typeof choice?.message.reasoning === "string" ? choice.message.reasoning : "");
  if (reasoning && reasoning.trim()) {
    blocks.push({ type: "thinking", thinking: reasoning.trim() });
  }
  if (choice?.message.content) {
    // Some models still inline <think>…</thinking> tags inside content; split
    // those out into a thinking block as well.
    const split = splitThinking(choice.message.content);
    if (split.thinking) blocks.push({ type: "thinking", thinking: split.thinking });
    if (split.text) blocks.push({ type: "text", text: split.text });
  }
  if (choice?.message.tool_calls) {
    for (const tc of choice.message.tool_calls) {
      let input: Record<string, unknown> = {};
      try { input = JSON.parse(tc.function.arguments || "{}"); } catch { /* ignore */ }
      blocks.push({ type: "tool_use", id: tc.id, name: tc.function.name, input });
    }
  }
  if (blocks.length === 0) blocks.push({ type: "text", text: "" });

  return {
    id: resp.id || `msg_${Date.now()}`,
    type: "message",
    role: "assistant",
    model: requestedModel,
    content: blocks,
    stop_reason: mapStopReason(choice?.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: resp.usage?.prompt_tokens ?? 0,
      output_tokens: resp.usage?.completion_tokens ?? 0,
    },
  };
}
