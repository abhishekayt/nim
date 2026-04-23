import type { Response } from "express";
import { ThinkingStreamParser } from "./thinking";

interface DeltaToolCall {
  index: number;
  id?: string;
  type?: "function";
  function?: { name?: string; arguments?: string };
}

interface OpenAIStreamChunk {
  id?: string;
  model?: string;
  choices?: Array<{
    index: number;
    delta?: {
      role?: string;
      content?: string | null;
      tool_calls?: DeltaToolCall[];
    };
    finish_reason?: string | null;
  }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
}

function sseSend(res: Response, event: string, data: unknown) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function mapStopReason(r: string | null | undefined): "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" {
  switch (r) {
    case "length": return "max_tokens";
    case "tool_calls": return "tool_use";
    default: return "end_turn";
  }
}

/**
 * Translate an OpenAI SSE stream from NIM into Anthropic Messages SSE events,
 * writing them to `res`. Resolves when the stream ends.
 */
export async function streamOpenAIToAnthropic(opts: {
  upstream: ReadableStream<Uint8Array>;
  res: Response;
  requestedModel: string;
}): Promise<void> {
  const { upstream, res, requestedModel } = opts;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const messageId = `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  let messageStarted = false;
  let textBlockOpen = false;
  let thinkingBlockOpen = false;
  const toolBlocks = new Map<number, { anthropicIndex: number; id: string; name: string; argsBuffer: string }>();
  let nextBlockIndex = 0;
  let textBlockIndex = -1;
  let thinkingBlockIndex = -1;
  let stopReason: string | null = null;
  let usage = { input_tokens: 0, output_tokens: 0 };
  const thinkParser = new ThinkingStreamParser();

  const startMessage = () => {
    if (messageStarted) return;
    messageStarted = true;
    sseSend(res, "message_start", {
      type: "message_start",
      message: {
        id: messageId,
        type: "message",
        role: "assistant",
        model: requestedModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  };

  const openTextBlock = () => {
    if (textBlockOpen) return;
    closeThinkingBlock();
    startMessage();
    textBlockIndex = nextBlockIndex++;
    textBlockOpen = true;
    sseSend(res, "content_block_start", {
      type: "content_block_start",
      index: textBlockIndex,
      content_block: { type: "text", text: "" },
    });
  };

  const closeTextBlock = () => {
    if (!textBlockOpen) return;
    sseSend(res, "content_block_stop", { type: "content_block_stop", index: textBlockIndex });
    textBlockOpen = false;
  };

  const openThinkingBlock = () => {
    if (thinkingBlockOpen) return;
    startMessage();
    thinkingBlockIndex = nextBlockIndex++;
    thinkingBlockOpen = true;
    sseSend(res, "content_block_start", {
      type: "content_block_start",
      index: thinkingBlockIndex,
      content_block: { type: "thinking", thinking: "" },
    });
  };

  const closeThinkingBlock = () => {
    if (!thinkingBlockOpen) return;
    sseSend(res, "content_block_stop", { type: "content_block_stop", index: thinkingBlockIndex });
    thinkingBlockOpen = false;
  };

  const emitContent = (raw: string) => {
    if (!raw) return;
    const { thinking, text } = thinkParser.feed(raw);
    if (thinking) {
      openThinkingBlock();
      sseSend(res, "content_block_delta", {
        type: "content_block_delta",
        index: thinkingBlockIndex,
        delta: { type: "thinking_delta", thinking },
      });
    }
    if (text) {
      openTextBlock();
      sseSend(res, "content_block_delta", {
        type: "content_block_delta",
        index: textBlockIndex,
        delta: { type: "text_delta", text },
      });
    }
  };

  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || !line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;

        let chunk: OpenAIStreamChunk;
        try { chunk = JSON.parse(payload); } catch { continue; }

        if (chunk.usage) {
          usage = {
            input_tokens: chunk.usage.prompt_tokens ?? usage.input_tokens,
            output_tokens: chunk.usage.completion_tokens ?? usage.output_tokens,
          };
        }

        const choice = chunk.choices?.[0];
        if (!choice) continue;
        const delta = choice.delta ?? {};

        if (typeof delta.content === "string" && delta.content.length > 0) {
          emitContent(delta.content);
        }

        if (delta.tool_calls) {
          closeTextBlock();
          closeThinkingBlock();
          for (const tc of delta.tool_calls) {
            let entry = toolBlocks.get(tc.index);
            if (!entry) {
              startMessage();
              const anthropicIndex = nextBlockIndex++;
              entry = {
                anthropicIndex,
                id: tc.id ?? `toolu_${Date.now()}_${tc.index}`,
                name: tc.function?.name ?? "",
                argsBuffer: "",
              };
              toolBlocks.set(tc.index, entry);
              sseSend(res, "content_block_start", {
                type: "content_block_start",
                index: anthropicIndex,
                content_block: { type: "tool_use", id: entry.id, name: entry.name, input: {} },
              });
            } else if (tc.id && !entry.id) {
              entry.id = tc.id;
            }
            if (tc.function?.name && !entry.name) entry.name = tc.function.name;
            const argsPiece = tc.function?.arguments ?? "";
            if (argsPiece) {
              entry.argsBuffer += argsPiece;
              sseSend(res, "content_block_delta", {
                type: "content_block_delta",
                index: entry.anthropicIndex,
                delta: { type: "input_json_delta", partial_json: argsPiece },
              });
            }
          }
        }

        if (choice.finish_reason) stopReason = choice.finish_reason;
      }
    }
  } finally {
    // Flush any buffered text/thinking the parser is still holding
    const tail = thinkParser.end();
    if (tail.thinking) {
      openThinkingBlock();
      sseSend(res, "content_block_delta", {
        type: "content_block_delta",
        index: thinkingBlockIndex,
        delta: { type: "thinking_delta", thinking: tail.thinking },
      });
    }
    if (tail.text) {
      openTextBlock();
      sseSend(res, "content_block_delta", {
        type: "content_block_delta",
        index: textBlockIndex,
        delta: { type: "text_delta", text: tail.text },
      });
    }
    closeTextBlock();
    closeThinkingBlock();
    for (const entry of toolBlocks.values()) {
      sseSend(res, "content_block_stop", { type: "content_block_stop", index: entry.anthropicIndex });
    }
    if (messageStarted) {
      sseSend(res, "message_delta", {
        type: "message_delta",
        delta: { stop_reason: mapStopReason(stopReason), stop_sequence: null },
        usage: { output_tokens: usage.output_tokens },
      });
      sseSend(res, "message_stop", { type: "message_stop" });
    }
    res.end();
  }
}
