import type { AnthropicRequest, AnthropicBlock } from "./translator";
import type { ModelCategory } from "./store";

/**
 * Inspect an Anthropic-format request and return an ordered list of model
 * categories the router should prefer. The first category is the strongest
 * preference; later entries are fallbacks.
 *
 * Rules (cheap, deterministic, no model calls):
 *   - any image block → "vision" (hard requirement)
 *   - tools present + recent file/code signals → "coding"
 *   - explicit reasoning verbs ("think", "plan", "design", "analyze") or a
 *     long context (>8k chars) without coding signals → "reasoning"
 *   - otherwise → "general"
 */
export function classifyRequest(req: AnthropicRequest): {
  categories: ModelCategory[];
  signals: string[];
} {
  const signals: string[] = [];
  let hasImage = false;
  let lastUserText = "";
  let totalChars = 0;

  for (const m of req.messages) {
    const blocks: AnthropicBlock[] = typeof m.content === "string"
      ? [{ type: "text", text: m.content }]
      : m.content;
    for (const b of blocks) {
      if (b.type === "image") hasImage = true;
      else if (b.type === "text") {
        totalChars += b.text.length;
        if (m.role === "user") lastUserText = b.text;
      } else if (b.type === "tool_result") {
        const txt = typeof b.content === "string"
          ? b.content
          : Array.isArray(b.content) ? b.content.map((c) => c.text).join("\n") : "";
        totalChars += txt.length;
      }
    }
  }

  const sysText = typeof req.system === "string"
    ? req.system
    : Array.isArray(req.system) ? req.system.map((s) => s.text).join("\n") : "";
  totalChars += sysText.length;

  if (hasImage) {
    signals.push("image-block");
    return { categories: ["vision", "general", "reasoning", "coding"], signals };
  }

  const tools = req.tools ?? [];
  const toolNames = new Set(tools.map((t) => t.name.toLowerCase()));
  const hasCodingTools = ["edit", "write", "read", "bash", "str_replace", "create_file", "view"]
    .some((n) => Array.from(toolNames).some((tn) => tn.includes(n)));

  const lastLower = lastUserText.toLowerCase();
  const codingHints = /\b(refactor|implement|fix bug|stack trace|compile|typescript|javascript|python|rust|golang?|function|class\s+\w+|import\s+|const\s+\w+|def\s+\w+|\.tsx?|\.jsx?|\.py|\.rs|\.go|file:|line\s*\d+)\b/i.test(lastUserText)
    || /```/.test(lastUserText);
  const reasoningHints = /\b(think|plan|design|analy[sz]e|why|how does|explain|architecture|trade.?off|approach|strategy)\b/i.test(lastLower);
  const planMode = /plan\s*mode|planning\s*phase/i.test(sysText);

  if (hasCodingTools) signals.push("coding-tools");
  if (codingHints) signals.push("coding-hints");
  if (reasoningHints) signals.push("reasoning-hints");
  if (planMode) signals.push("plan-mode");
  if (totalChars > 8000) signals.push("long-context");

  if (planMode || (reasoningHints && !codingHints && !hasCodingTools)) {
    return { categories: ["reasoning", "coding", "general"], signals };
  }
  if (hasCodingTools || codingHints) {
    return { categories: ["coding", "general", "reasoning"], signals };
  }
  if (totalChars > 8000) {
    return { categories: ["reasoning", "general", "coding"], signals };
  }
  return { categories: ["general", "coding", "reasoning"], signals };
}
