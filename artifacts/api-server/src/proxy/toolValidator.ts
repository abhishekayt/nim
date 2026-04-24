/**
 * Tool use reliability engine.
 *
 * Validates tool call JSON against the actual JSON Schema using Ajv,
 * provides a multi-pass auto-repair loop, and exposes helpers for the
 * proxy retry path.
 *
 * Failure modes we handle (in order of severity / cost to fix):
 *   1. Invalid JSON syntax (trailing commas, comments, unquoted keys).
 *   2. Missing required fields.
 *   3. Wrong types (string where number expected, etc.) — coerced when
 *      safe, otherwise reported.
 *   4. Tool call printed as prose instead of via the function-calling
 *      API — extracted by `extractToolCallsFromText`.
 *   5. Hallucinated tool name not in the available schema set.
 */

import Ajv, { type ErrorObject, type ValidateFunction } from "ajv";
import addFormats from "ajv-formats";
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

// ---- Ajv setup ----
// Strict mode off so unfamiliar keywords (e.g. "uniqueKeyword") don't blow
// up. Coerce types when safe (string→number etc.) — this matches what
// most well-behaved tool callers do server-side.
let ajvInstance: Ajv | null = null;

function getAjv(): Ajv {
  if (ajvInstance) return ajvInstance;
  const ajv = new Ajv({
    strict: false,
    allErrors: true,
    coerceTypes: true,
    useDefaults: true,
    removeAdditional: false,
  });
  addFormats(ajv);
  ajvInstance = ajv;
  return ajv;
}

const schemaCache = new WeakMap<object, ValidateFunction>();

function compileSchema(parameters: Record<string, unknown>): ValidateFunction | null {
  if (!parameters || typeof parameters !== "object") return null;
  const cached = schemaCache.get(parameters);
  if (cached) return cached;
  try {
    const validate = getAjv().compile(parameters);
    schemaCache.set(parameters, validate);
    return validate;
  } catch {
    return null;
  }
}

function describeAjvErrors(errors: readonly ErrorObject[] | null | undefined): string[] {
  if (!errors || errors.length === 0) return [];
  return errors.slice(0, 8).map((e) => {
    const where = e.instancePath || "(root)";
    return `${where} ${e.message ?? "invalid"}`;
  });
}

// ---- Structural JSON validation (cheap pre-check) ----
export function validateJsonStructure(jsonStr: string): ValidationResult {
  const errors: string[] = [];
  if (!jsonStr || jsonStr.trim() === "") {
    return { valid: false, errors: ["Empty arguments string"] };
  }
  if (/,(\s*[}\]])/g.test(jsonStr)) errors.push("Trailing comma detected");
  if (/\/\/.*\n|\/\*[\s\S]*?\*\//.test(jsonStr)) errors.push("Comments detected in JSON");
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
  if (JSON.stringify(parsed).includes("undefined")) {
    errors.push("Contains undefined values");
  }
  return { valid: errors.length === 0, errors };
}

export function fixJsonErrors(jsonStr: string): string | null {
  if (!jsonStr) return null;
  let fixed = jsonStr;

  // Remove ``` fences if present.
  fixed = fixed.replace(/^```(?:json|tool[_ ]?call)?\s*/im, "").replace(/\s*```\s*$/m, "");

  // Trailing commas.
  fixed = fixed.replace(/,(\s*[}\]])/g, "$1");
  // Comments.
  fixed = fixed.replace(/\/\/.*$/gm, "");
  fixed = fixed.replace(/\/\*[\s\S]*?\*\//g, "");
  // Unquoted keys (heuristic).
  fixed = fixed.replace(/([{,]\s*)([a-zA-Z_]\w*)(\s*:)/g, '$1"$2"$3');
  // Single-quoted strings → double-quoted.
  fixed = fixed.replace(/'([^'\\]*(?:\\.[^'\\]*)*)'(\s*[,:}\]])/g, '"$1"$2');
  // Smart quotes (some models emit \u201c\u201d).
  fixed = fixed.replace(/[\u201C\u201D]/g, '"').replace(/[\u2018\u2019]/g, "'");

  try {
    JSON.parse(fixed);
    return fixed;
  } catch {
    return null;
  }
}

/**
 * Type-coercion repair: walk a parsed args object against the schema and
 * fix simple type mismatches (e.g. "5" → 5 for a numeric field).
 */
function coerceTypes(
  args: Record<string, unknown>,
  schema: Record<string, unknown> | null,
): Record<string, unknown> {
  if (!schema || schema["type"] !== "object") return args;
  const props = (schema["properties"] as Record<string, { type?: string | string[] }> | undefined) ?? {};
  const out: Record<string, unknown> = { ...args };
  for (const [k, v] of Object.entries(args)) {
    const prop = props[k];
    if (!prop) continue;
    const t = Array.isArray(prop.type) ? prop.type[0] : prop.type;
    if (t === "number" || t === "integer") {
      if (typeof v === "string" && /^-?\d+(\.\d+)?$/.test(v.trim())) {
        out[k] = t === "integer" ? parseInt(v, 10) : parseFloat(v);
      }
    } else if (t === "boolean") {
      if (v === "true") out[k] = true;
      else if (v === "false") out[k] = false;
    } else if (t === "string") {
      if (typeof v === "number" || typeof v === "boolean") {
        out[k] = String(v);
      }
    } else if (t === "array") {
      if (typeof v === "string") {
        // Try comma-split as a last resort.
        const trimmed = v.trim();
        if (trimmed.startsWith("[")) {
          try {
            const parsed = JSON.parse(trimmed);
            if (Array.isArray(parsed)) out[k] = parsed;
          } catch {
            /* ignore */
          }
        }
      }
    }
  }
  return out;
}

/**
 * Fill in safe defaults for missing required string fields when the
 * schema doesn't specify a default.
 */
function fillEmptyRequiredStrings(
  args: Record<string, unknown>,
  schema: Record<string, unknown> | null,
): Record<string, unknown> {
  if (!schema) return args;
  const required = (schema["required"] as string[] | undefined) ?? [];
  const props = (schema["properties"] as Record<string, { type?: string | string[]; default?: unknown }> | undefined) ?? {};
  const out: Record<string, unknown> = { ...args };
  for (const r of required) {
    if (r in out && out[r] !== undefined && out[r] !== null) continue;
    const prop = props[r];
    if (!prop) continue;
    if ("default" in prop) {
      out[r] = prop.default;
      continue;
    }
    const t = Array.isArray(prop.type) ? prop.type[0] : prop.type;
    if (t === "string") out[r] = "";
    // Don't fabricate numbers/booleans — that would be guessing.
  }
  return out;
}

/**
 * Main entry: try to repair the arguments string for a single tool call
 * against its schema. Returns the (possibly modified) args string and a
 * list of repairs applied.
 */
export function repairToolCallArgs(
  argsStr: string,
  schema: Record<string, unknown> | null,
): { argsStr: string; repaired: boolean; notes: string[] } {
  const notes: string[] = [];

  // Pass 1: JSON structural heal.
  let parsed: Record<string, unknown> | null = null;
  try {
    parsed = JSON.parse(argsStr || "{}");
  } catch {
    const healed = fixJsonErrors(argsStr);
    if (healed) {
      try {
        parsed = JSON.parse(healed);
        notes.push("JSON syntax repaired");
      } catch {
        return { argsStr, repaired: false, notes: ["unparseable JSON"] };
      }
    } else {
      return { argsStr, repaired: false, notes: ["unparseable JSON"] };
    }
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { argsStr, repaired: false, notes: ["not an object"] };
  }

  // Pass 2: type coercion.
  const before = JSON.stringify(parsed);
  parsed = coerceTypes(parsed, schema);
  if (JSON.stringify(parsed) !== before) notes.push("types coerced");

  // Pass 3: fill empty required strings.
  const beforeFill = JSON.stringify(parsed);
  parsed = fillEmptyRequiredStrings(parsed, schema);
  if (JSON.stringify(parsed) !== beforeFill) notes.push("required fields defaulted");

  return {
    argsStr: JSON.stringify(parsed),
    repaired: notes.length > 0,
    notes,
  };
}

/**
 * Validate all tool calls in a response against their schemas using Ajv.
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

    if (!schema) {
      return {
        valid: false,
        errors: [`Tool ${tc.function.name} (index ${i}): unknown tool name (not in available schemas)`],
        toolIndex: i,
      };
    }

    // Structural JSON.
    const struct = validateJsonStructure(tc.function.arguments || "{}");
    if (!struct.valid) {
      return {
        valid: false,
        errors: [`Tool ${tc.function.name} (index ${i}): ${struct.errors.join(", ")}`],
        toolIndex: i,
      };
    }

    let argsObj: unknown;
    try {
      argsObj = JSON.parse(tc.function.arguments || "{}");
    } catch {
      return {
        valid: false,
        errors: [`Tool ${tc.function.name} (index ${i}): JSON parse failed`],
        toolIndex: i,
      };
    }

    const validate = compileSchema(schema.parameters);
    if (validate) {
      const ok = validate(argsObj);
      if (!ok) {
        return {
          valid: false,
          errors: [
            `Tool ${tc.function.name} (index ${i}): ${describeAjvErrors(validate.errors).join("; ")}`,
          ],
          toolIndex: i,
        };
      }
    } else {
      // Schema couldn't compile; fall back to required-only check.
      const required = (schema.parameters?.["required"] as string[] | undefined) ?? [];
      const obj = argsObj as Record<string, unknown>;
      const missing = required.filter((r) => !(r in obj));
      if (missing.length > 0) {
        return {
          valid: false,
          errors: [`Tool ${tc.function.name}: missing required fields: ${missing.join(", ")}`],
          toolIndex: i,
        };
      }
    }
  }

  return { valid: true, errors: [] };
}

/**
 * Try to repair every tool call in a response in-place. Returns the
 * (possibly mutated) response and a summary of what was repaired.
 */
export function repairResponseToolCalls(
  data: OpenAINonStreamResponse,
  toolSchemas: ToolSchema[],
): { data: OpenAINonStreamResponse; repaired: boolean; notes: string[] } {
  const choice = data.choices?.[0];
  if (!choice?.message?.tool_calls?.length) {
    return { data, repaired: false, notes: [] };
  }
  const schemaMap = new Map(toolSchemas.map((s) => [s.name, s]));
  const cloned: OpenAINonStreamResponse = JSON.parse(JSON.stringify(data));
  const tcs = cloned.choices[0]!.message.tool_calls!;
  const allNotes: string[] = [];
  let any = false;
  for (let i = 0; i < tcs.length; i++) {
    const tc = tcs[i]!;
    const schema = schemaMap.get(tc.function.name);
    const params = (schema?.parameters as Record<string, unknown> | undefined) ?? null;
    const r = repairToolCallArgs(tc.function.arguments || "{}", params);
    if (r.repaired) {
      tc.function.arguments = r.argsStr;
      any = true;
      allNotes.push(`tool ${tc.function.name}: ${r.notes.join(", ")}`);
    }
  }
  return { data: cloned, repaired: any, notes: allNotes };
}

/**
 * Detect malformed tool calls. Used by the proxy retry decision.
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

  for (const tc of tcs) {
    const v = validateJsonStructure(tc.function.arguments || "{}");
    if (!v.valid) return true;
  }

  if (toolSchemas && toolSchemas.length > 0) {
    const v = validateToolCalls(data, toolSchemas);
    if (!v.valid) return true;
  }

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
    const looksLikeXmlParams = /<tool>\s*\w+\s*<\/tool>\s*<[\w_]+>/.test(txt);
    if (looksLikeXmlParams) return true;
  }

  return false;
}

/**
 * Extract a tool call from prose content (model "described" the call).
 */
export function extractToolCallsFromText(
  data: OpenAINonStreamResponse,
): OpenAINonStreamResponse | null {
  const choice = data.choices?.[0];
  if (!choice) return null;

  const txt = (choice.message?.content ?? "").trim();
  if (!txt) return null;

  const candidateBlobs: string[] = [];
  const fenceMatch = txt.match(/```(?:json|tool)?\s*([\s\S]*?)```/);
  if (fenceMatch) candidateBlobs.push(fenceMatch[1]!.trim());
  const objMatch = txt.match(/\{[\s\S]*"name"\s*:\s*"[^"]+"[\s\S]*\}/);
  if (objMatch) candidateBlobs.push(objMatch[0]);

  for (const blob of candidateBlobs) {
    let parsed: Record<string, unknown> | null = null;
    try {
      parsed = JSON.parse(blob) as Record<string, unknown>;
    } catch {
      const healed = fixJsonErrors(blob);
      if (healed) {
        try {
          parsed = JSON.parse(healed) as Record<string, unknown>;
        } catch {
          continue;
        }
      } else continue;
    }
    if (!parsed?.name) continue;
    const fixed: OpenAINonStreamResponse = JSON.parse(JSON.stringify(data));
    const c = fixed.choices[0]!;
    c.message.content = null;
    c.message.tool_calls = [{
      id: `call_${Date.now()}`,
      type: "function",
      function: {
        name: String(parsed.name),
        arguments: JSON.stringify(parsed["arguments"] ?? parsed["input"] ?? parsed["parameters"] ?? {}),
      },
    }];
    return fixed;
  }

  return null;
}

/**
 * Build an escalating self-correction prompt for retrying malformed
 * tool calls. The intensity ramps with retry count.
 */
export function buildToolCorrectionPrompt(
  originalResponse: OpenAINonStreamResponse,
  errors: string[],
  attempt: number,
): string {
  const choice = originalResponse.choices?.[0];
  const toolCalls = choice?.message?.tool_calls ?? [];

  const headline =
    attempt <= 1
      ? "Your previous tool call response was invalid."
      : attempt === 2
        ? "Your retry was STILL invalid. Read the schema carefully this time."
        : "FINAL ATTEMPT. Your previous tool calls keep failing schema validation.";

  let prompt = `${headline}\nErrors:\n`;
  for (const err of errors) prompt += `- ${err}\n`;

  if (toolCalls.length > 0) {
    prompt += `\nYour previous tool calls (these are wrong):\n`;
    for (const tc of toolCalls) {
      prompt += `- ${tc.function.name}: ${tc.function.arguments}\n`;
    }
  }

  prompt += `\nRequirements:\n`;
  prompt += `- Use the function-calling API only — NOT JSON in prose, NOT markdown fences.\n`;
  prompt += `- Every required field must be present and the correct type.\n`;
  prompt += `- No trailing commas, no comments, no single quotes, all keys quoted.\n`;
  if (attempt >= 2) {
    prompt += `- If you cannot satisfy the schema, respond in plain prose explaining what is missing instead of guessing.\n`;
  }
  return prompt;
}
