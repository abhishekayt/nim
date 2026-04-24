/**
 * Smart classifier with LLM-based fallback.
 *
 * When the regex classifier has low confidence, this module uses a cheap
 * LLM call to a small model to determine the best model category.
 * This dramatically improves routing accuracy for ambiguous requests.
 */

import type { AnthropicRequest, AnthropicBlock } from "./translator";
import type { ModelCategory } from "./store";
import { classifyRequestWithConfidence } from "./classifier";

interface ClassificationResult {
  categories: ModelCategory[];
  signals: string[];
  confidence: number;
  method: "regex" | "llm";
}

const CONFIDENCE_THRESHOLD = 0.6;

function extractRequestSummary(req: AnthropicRequest): string {
  const parts: string[] = [];

  // Extract system prompt hints
  const sysText = typeof req.system === "string"
    ? req.system
    : Array.isArray(req.system) ? req.system.map((s) => s.text).join("\n") : "";
  if (sysText) parts.push(`System: ${sysText.slice(0, 500)}`);

  // Extract last user message
  let lastUserText = "";
  let totalChars = 0;
  for (const m of req.messages) {
    const blocks: AnthropicBlock[] = typeof m.content === "string"
      ? [{ type: "text", text: m.content }]
      : m.content;
    for (const b of blocks) {
      if (b.type === "text") {
        totalChars += b.text.length;
        if (m.role === "user") lastUserText = b.text;
      }
    }
  }

  if (lastUserText) parts.push(`User: ${lastUserText.slice(0, 2000)}`);
  if (totalChars > 8000) parts.push(`(Long context: ${Math.round(totalChars / 4)} estimated tokens)`);

  // Tool hints
  const tools = req.tools ?? [];
  if (tools.length > 0) {
    const toolNames = tools.map((t) => t.name).join(", ");
    parts.push(`Tools: ${toolNames}`);
  }

  return parts.join("\n---\n");
}

const CLASSIFICATION_PROMPT = `You are a request classifier for an AI model router. Your job is to categorize incoming requests into one of these categories, ordered by preference:

- "coding": Code writing, debugging, refactoring, file editing, build tasks, app creation
- "reasoning": Deep analysis, planning, architecture, math, logic puzzles, long-context synthesis
- "vision": Image understanding, visual analysis, screenshots, diagrams
- "general": Everything else — chat, Q&A, creative writing, summaries

Rules:
1. If the request contains images, vision MUST be first.
2. If the request involves writing/editing code or has developer tools, coding should be first.
3. If the request asks for analysis, planning, or complex reasoning, reasoning should be first.
4. Otherwise, general is first.

Respond ONLY with a JSON object in this exact format:
{"categories": ["coding"|"reasoning"|"vision"|"general", ...], "confidence": 0.0-1.0, "reasoning": "brief explanation"}

The categories array should have all 4 categories in preference order.`;

async function callClassifierLLM(summary: string): Promise<ClassificationResult | null> {
  const apiKey = process.env["NIM_API_KEY_1"] || process.env["NIM_API_KEY"];
  const baseUrl = process.env["NIM_CLASSIFIER_URL"] || "https://integrate.api.nvidia.com/v1";
  const model = process.env["NIM_CLASSIFIER_MODEL"] || "meta/llama-3.2-3b-instruct";

  if (!apiKey) return null;

  try {
    const res = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: CLASSIFICATION_PROMPT },
          { role: "user", content: summary },
        ],
        temperature: 0.1,
        max_tokens: 256,
        response_format: { type: "json_object" },
      }),
    });

    if (!res.ok) return null;

    const data = await res.json() as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const content = data.choices?.[0]?.message?.content;
    if (!content) return null;

    const parsed = JSON.parse(content) as {
      categories?: string[];
      confidence?: number;
      reasoning?: string;
    };

    const validCategories: ModelCategory[] = ["coding", "reasoning", "vision", "general"];
    const categories = (parsed.categories ?? [])
      .filter((c): c is ModelCategory => validCategories.includes(c as ModelCategory));

    if (categories.length === 0) return null;

    // Ensure all categories are present
    for (const cat of validCategories) {
      if (!categories.includes(cat)) categories.push(cat);
    }

    return {
      categories,
      signals: ["llm-classified", parsed.reasoning ? `llm-reason:${parsed.reasoning.slice(0, 50)}` : "llm-classified"],
      confidence: Math.min(Math.max(parsed.confidence ?? 0.7, 0), 1),
      method: "llm",
    };
  } catch {
    return null;
  }
}

/**
 * Classify a request using regex first, then fall back to LLM if confidence is low.
 */
export async function classifyRequestSmart(req: AnthropicRequest): Promise<ClassificationResult> {
  const regexResult = classifyRequestWithConfidence(req);

  // High-confidence regex results are trusted immediately
  if (regexResult.confidence >= CONFIDENCE_THRESHOLD) {
    return {
      categories: regexResult.categories,
      signals: regexResult.signals,
      confidence: regexResult.confidence,
      method: "regex",
    };
  }

  // Low confidence: try LLM fallback
  const summary = extractRequestSummary(req);
  const llmResult = await callClassifierLLM(summary);

  if (llmResult) {
    return {
      ...llmResult,
      signals: [...regexResult.signals, ...llmResult.signals],
    };
  }

  // LLM failed, return regex result anyway
  return {
    categories: regexResult.categories,
    signals: regexResult.signals,
    confidence: regexResult.confidence,
    method: "regex",
  };
}
