/**
 * Thinking-block utilities. Reasoning models like DeepSeek-R1 emit their
 * chain-of-thought wrapped in `<think>…</think>` tags before the actual
 * answer. We extract those into Anthropic-style "thinking" content blocks
 * so Claude Code displays them in its dedicated thinking panel instead of
 * polluting the normal message body.
 */

export interface ThinkingSplit {
  thinking: string | null;
  text: string;
}

const OPEN = /<think(?:ing)?>/i;
const CLOSE = /<\/think(?:ing)?>/i;

export function splitThinking(content: string): ThinkingSplit {
  if (!content) return { thinking: null, text: "" };
  const open = content.match(OPEN);
  if (!open || open.index === undefined) return { thinking: null, text: content };
  const after = content.slice(open.index + open[0].length);
  const close = after.match(CLOSE);
  if (!close || close.index === undefined) {
    // Unclosed → treat the rest as thinking, no answer yet
    return { thinking: after.trim(), text: "" };
  }
  const thinking = after.slice(0, close.index).trim();
  const text = (content.slice(0, open.index) + after.slice(close.index + close[0].length)).trim();
  return { thinking: thinking || null, text };
}

/**
 * Streaming thinking parser: feed it text chunks and it tells you which
 * portion belongs to a thinking block vs the user-visible text. Handles
 * the open/close tags arriving split across SSE chunks.
 */
export class ThinkingStreamParser {
  private state: "pre" | "thinking" | "post" = "pre";
  private buffer = "";

  feed(chunk: string): { thinking: string; text: string } {
    let outThinking = "";
    let outText = "";
    this.buffer += chunk;

    // Loop because state may flip multiple times in one buffer
    // (extremely rare but defended against).
    while (true) {
      if (this.state === "pre") {
        const m = this.buffer.match(OPEN);
        if (m && m.index !== undefined) {
          // Anything before the tag is plain text (rare for R1 but possible)
          outText += this.buffer.slice(0, m.index);
          this.buffer = this.buffer.slice(m.index + m[0].length);
          this.state = "thinking";
          continue;
        }
        // No open tag yet: hold back the last few chars in case the tag
        // is split across chunks (e.g. "<thi" + "nk>").
        if (this.buffer.length > 16) {
          outText += this.buffer.slice(0, this.buffer.length - 16);
          this.buffer = this.buffer.slice(-16);
        }
        break;
      }
      if (this.state === "thinking") {
        const m = this.buffer.match(CLOSE);
        if (m && m.index !== undefined) {
          outThinking += this.buffer.slice(0, m.index);
          this.buffer = this.buffer.slice(m.index + m[0].length);
          this.state = "post";
          continue;
        }
        if (this.buffer.length > 16) {
          outThinking += this.buffer.slice(0, this.buffer.length - 16);
          this.buffer = this.buffer.slice(-16);
        }
        break;
      }
      // post: everything is plain text
      outText += this.buffer;
      this.buffer = "";
      break;
    }
    return { thinking: outThinking, text: outText };
  }

  /** Flush remaining buffer at end of stream. */
  end(): { thinking: string; text: string } {
    let thinking = "";
    let text = "";
    if (this.state === "pre" || this.state === "post") text = this.buffer;
    else thinking = this.buffer;
    this.buffer = "";
    return { thinking, text };
  }

  hasEmittedThinking(): boolean {
    return this.state !== "pre";
  }
}
