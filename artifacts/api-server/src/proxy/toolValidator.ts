/**
 * Tool use reliability engine.
 *
 * Validates tool call JSON against schemas, provides self-correction loops,
 * and handles common failure modes from open-weight models.
 */

import type { OpenAIToolCall, OpenAINonStreamResponse } from "./translator";

interface ToolSchema {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
  fixed?: OpenAIToolCall;
}

/**
 * Quick structural JSON validation without a full schema validator.
 * Checks for:
 * - Valid JSON parse
 * - No trailing commas
 * - Required top-level object
 * - No undefined values
 */
export function validateJsonStructure(jsonStr: string): ValidationResult {
  const errors: string[] = [];

  if (!jsonStr || jsonStr.trim() === "") {
    return { valid: false, errors: ["Empty arguments string"] };
  }

  // Check for trailing commas (common failure mode)
  if (/,(\s*[}\]])/g.test(jsonStr)) {
    errors.push("Trailing comma detected");
  }

  // Check for comments in JSON (some models emit these)
  if (/\/\/.*\n|\/\*[\s\S]*?\*\//.test(jsonStr)) {
    errors.push("Comments detected in JSON");
  }

  // Check for unquoted keys (common with some models)
  if (/{\s*[a-zA-Z_]\w*\s*:/.test(jsonStr) && !/"[a-zA-Z_]\w*"\s*:/.test(jsonStr)) {
    errors.push("Potentially unquoted keys");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonStr);
  } catch (e) {
    errors.push(`Parse error: ${e instanceof Error ? e.message : String(e)}`);
    return { valid: false, errors };
  }

  if (typeof parsed !== "object" || parsed === null) {
    errors.push("Arguments must be a JSON object");
    return { valid: false, errors };
  }

  // Check for undefined-like values (null is fine, undefined is not valid JSON anyway)
  const str = JSON.stringify(parsed);
  if (str.includes("undefined")) {
    errors.push("Contains undefined values");
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Attempt to fix common JSON errors from model outputs.
 */
export function fixJsonErrors(jsonStr: string): string | null {
  if (!jsonStr) return null;

  let fixed = jsonStr;

  // Remove trailing commas before } or ]
  fixed = fixed.replace(/,(\s*[}\]])/g, "$1");

  // Remove comments
  fixed = fixed.replace(/\/\/.*$/gm, "");
  fixed = fixed.replace(/\/\*[\s\S]*?\*\//g, "");

  // Quote unquoted keys (simple heuristic)
  fixed = fixed.replace(/([{,]\s*)([a-zA-Z_]\w*)(\s*:)/g, '$1"$2"$3');

  // Remove extra whitespace
  fixed = fixed.replace(/\n\s*/g, " ");

  try {
    JSON.parse(fixed);
    return fixed;
  } catch {
    return null;
  }
}

/**
 * Validate all tool calls in a response against their schemas.
 */
export function validateToolCalls(
  data: OpenAINonStreamResponse,
  toolSchemas: ToolSchema[],
): { valid: boolean; errors: string[]; toolIndex?: number } {
  const choice = data.choices?.[0];
  if (!choice) return { valid: false, errors: ["No choices in response"] };

  const tcs = choice.message?.tool_calls ?? [];
  if (tcs.length === 0) return { valid: true, errors: [] };

  const schemaMap = new Map(toolSchemas.map((s) => [s.name, s]));

  for (let i = 0; i < tcs.length; i++) {
    const tc = tcs[i]!;
    const schema = schemaMap.get(tc.function.name);

    // First: structural validation
    const struct = validateJsonStructure(tc.function.arguments || "{}");
    if (!struct.valid) {
      return {
        valid: false,
        errors: [`Tool ${tc.function.name} (index ${i}): ${struct.errors.join(", ")}`],
        toolIndex: i,
      };
    }

    // Second: if we have a schema, do basic required-field check
    if (schema?.parameters) {
      const required = (schema.parameters as Record<string, unknown>).required as string[] | undefined;
      if (required && required.length > 0) {
        try {
          const args = JSON.parse(tc.function.arguments || "{}") as Record<string, unknown>;
          const missing = required.filter((r) => !(r in args));
          if (missing.length > 0) {
            return {
              valid: false,
              errors: [`Tool ${tc.function.name}: missing required fields: ${missing.join(", ")}`],
              toolIndex: i,
            };
          }
        } catch {
          // Already validated structurally above
        }
      }
    }
  }

  return { valid: true, errors: [] };
}

/**
 * Check if a response contains malformed tool calls.
 * More comprehensive than the basic check in nim-client.ts.
 */
export function isMalformedToolCallEnhanced(
  data: OpenAINonStreamResponse,
  hadTools: boolean,
  toolSchemas?: ToolSchema[],
): boolean {
  if (!hadTools) return false;

  const choice = data.choices?.[0];
  if (!choice) return false;

  const tcs = choice.message?.tool_calls ?? [];

  // Case 1: Has tool_calls but invalid JSON
  for (const tc of tcs) {
    const v = validateJsonStructure(tc.function.arguments || "{}");
    if (!v.valid) return true;
  }

  // Case 2: Has tool_calls but schema validation fails
  if (toolSchemas && toolSchemas.length > 0) {
    const v = validateToolCalls(data, toolSchemas);
    if (!v.valid) return true;
  }

  // Case 3: No tool_calls but content looks like a tool call
  if (tcs.length === 0) {
    const txt = (choice.message?.content ?? "").trim();
    if (!txt) return false;

    const looksLikeJsonToolCall =
      /```\s*(json|tool[_ ]?call)?[\s\S]*"(name|tool|function)"\s*:[\s\S]*"(arguments|input|parameters)"/i.test(txt) ||
      /^\s*\{[\s\S]*"(name|tool|function)"\s*:[\s\S]*"(arguments|input|parameters)"[\s\S]*\}\s*$/.test(txt);
    if (looksLikeJsonToolCall) return true;

    const looksLikeXmlToolCall =
      /<!\s*(tool[_ ]?call|function[_ ]?call|tool|invoke|use[_ ]?tool)\b[^>]*>/i.test(txt);
    if (looksLikeXmlToolCall) return true;

    // DeepSeek/Qwen sometimes output <tool>name</tool><parameter>value</parameter>
    const looksLikeXmlParams =
      /<tool>\s*\w+\s*<\/tool>\s*<[\w_]+>/.test(txt);
    if (looksLikeXmlParams) return true;
  }

  return false;
}

/**
 * Attempt to extract and fix tool calls from malformed text content.
 * Returns a modified response if extraction succeeds.
 */
export function extractToolCallsFromText(
  data: OpenAINonStreamResponse,
): OpenAINonStreamResponse | null {
  const choice = data.choices?.[0];
  if (!choice) return null;

  const txt = (choice.message?.content ?? "").trim();
  if (!txt) return null;

  // Try to extract JSON from markdown fences
  const fenceMatch = txt.match(/```(?:json|tool)?\s*([\s\S]*?)```/);
  if (fenceMatch) {
    try {
      const parsed = JSON.parse(fenceMatch[1]!.trim()) as Record<string, unknown>;
      if (parsed.name && (parsed.arguments || parsed.input || parsed.parameters)) {
        const fixed: OpenAINonStreamResponse = JSON.parse(JSON.stringify(data));
        const c = fixed.choices[0]!;
        c.message.content = null;
        c.message.tool_calls = [{
          id: `call_${Date.now()}`,
          type: "function",
          function: {
            name: String(parsed.name),
            arguments: JSON.stringify(parsed.arguments ?? parsed.input ?? parsed.parameters ?? {}),
          },
        }];
        return fixed;
      }
    } catch {
      // ignore
    }
  }

  // Try to extract raw JSON object
  const jsonMatch = txt.match(/\{[\s\S]*"name"\s*:\s*"[^"]+"[\s\S]*\}/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]) as Record<string, unknown>;
      if (parsed.name) {
        const fixed: OpenAINonStreamResponse = JSON.parse(JSON.stringify(data));
        const c = fixed.choices[0]!;
        c.message.content = null;
        c.message.tool_calls = [{
          id: `call_${Date.now()}`,
          type: "function",
          function: {
            name: String(parsed.name),
            arguments: JSON.stringify(parsed.arguments ?? parsed.input ?? parsed.parameters ?? {}),
          },
        }];
        return fixed;
      }
    } catch {
      // ignore
    }
  }

  return null;
}

/**
 * Build a self-correction prompt for retrying malformed tool calls.
 */
export function buildToolCorrectionPrompt(
  originalResponse: OpenAINonStreamResponse,
  errors: string[],
): string {
  const choice = originalResponse.choices?.[0];
  const toolCalls = choice?.message?.tool_calls ?? [];

  let prompt = `Your previous tool call response was invalid. Errors:\n`;
  for (const err of errors) {
    prompt += `- ${err}\n`;
  }

  if (toolCalls.length > 0) {
    prompt += `\nYour previous tool calls:\n`;
    for (const tc of toolCalls) {
      prompt += `- ${tc.function.name}: ${tc.function.arguments}\n`;
    }
  }

  prompt += `\nPlease retry using the function-calling API with valid JSON arguments. Ensure:\n`;
  prompt += `- All required fields are present\n`;
  prompt += `- No trailing commas\n`;
  prompt += `- No comments in JSON\n`;
  prompt += `- All keys are quoted\n`;

  return prompt;
}
