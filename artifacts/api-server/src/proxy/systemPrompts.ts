/**
 * Per-model system prompt engineering.
 *
 * Different model families respond differently to system prompt structure,
 * tool-use formatting, and reasoning instructions. These templates are
 * injected based on the resolved model name to maximize accuracy.
 */

export interface SystemPromptTemplate {
  /** Short identifier for the model family */
  family: string;
  /** Instructions prepended to the system prompt */
  preamble: string;
  /** Instructions appended after tool-use rules */
  postamble: string;
  /** Whether this model supports native reasoning/thinking mode */
  supportsThinking: boolean;
  /** How to activate thinking mode if supported */
  thinkingActivation?: string;
  /** Tool-use format preference */
  toolFormat: "openai" | "xml" | "auto";
  /** Max recommended tokens for this family */
  recommendedMaxTokens: number;
  /** Context window size */
  contextWindow: number;
}

const DEEPSEEK_TEMPLATE: SystemPromptTemplate = {
  family: "deepseek",
  preamble: `You are a helpful AI assistant. When reasoning through complex problems, take your time and think step by step before answering.

CRITICAL FORMATTING RULES:
- Use the function-calling API for all tool invocations. Do NOT wrap tool calls in markdown code blocks.
- Respond in plain text unless tool use is required.
- For code changes, make one focused edit at a time.`,
  postamble: `If you need to reason deeply about a problem, you may use your reasoning capability. Keep reasoning concise and directly relevant to the user's request.`,
  supportsThinking: true,
  thinkingActivation: "Enable deep reasoning mode for complex problems",
  toolFormat: "openai",
  recommendedMaxTokens: 8192,
  contextWindow: 128000,
};

const QWEN_TEMPLATE: SystemPromptTemplate = {
  family: "qwen",
  preamble: `You are Qwen, a helpful AI assistant created by Alibaba Cloud. You excel at coding, reasoning, and following instructions precisely.

IMPORTANT INSTRUCTIONS:
- When tools are available, invoke them via the function-calling API only.
- Never print tool calls as JSON text or wrap them in markdown fences.
- For coding tasks, prefer incremental changes over large rewrites.
- Validate that your tool call arguments match the schema exactly.`,
  postamble: `When solving complex problems, you may show your reasoning process. Keep it focused and brief.`,
  supportsThinking: true,
  thinkingActivation: "Use thinking mode for complex reasoning tasks",
  toolFormat: "openai",
  recommendedMaxTokens: 8192,
  contextWindow: 131072,
};

const KIMI_TEMPLATE: SystemPromptTemplate = {
  family: "kimi",
  preamble: `You are Kimi, an AI assistant by Moonshot AI. You are designed to be helpful, harmless, and honest.

TOOL USE GUIDELINES:
- Always use the function-calling API for tool invocations.
- Do NOT output tool calls as plain text JSON.
- Follow tool schemas exactly — required fields must be present.
- Make focused, minimal changes when editing files.`,
  postamble: `For complex tasks, reason step by step before producing your final answer.`,
  supportsThinking: true,
  thinkingActivation: "Activate reasoning for complex problems",
  toolFormat: "openai",
  recommendedMaxTokens: 8192,
  contextWindow: 256000,
};

const LLAMA_TEMPLATE: SystemPromptTemplate = {
  family: "llama",
  preamble: `You are a helpful AI assistant. You are direct, accurate, and follow instructions carefully.

CRITICAL RULES:
- Use function calling for all tool operations. Never output tool calls as text.
- When editing code, make small, focused changes.
- Ensure JSON arguments are valid and match the tool schema.
- Do not add trailing commas or comments inside JSON.`,
  postamble: `For difficult problems, think step by step before answering.`,
  supportsThinking: false,
  toolFormat: "openai",
  recommendedMaxTokens: 4096,
  contextWindow: 128000,
};

const GLM_TEMPLATE: SystemPromptTemplate = {
  family: "glm",
  preamble: `You are ChatGLM, a helpful AI assistant. You provide accurate, well-reasoned responses.

INSTRUCTIONS:
- Use the function-calling API for all tool invocations.
- Never wrap tool calls in markdown or print them as text.
- Make incremental, focused code changes.`,
  postamble: `Reason through complex problems step by step.`,
  supportsThinking: false,
  toolFormat: "openai",
  recommendedMaxTokens: 8192,
  contextWindow: 128000,
};

const NEMOTRON_TEMPLATE: SystemPromptTemplate = {
  family: "nemotron",
  preamble: `You are a helpful AI assistant optimized for reasoning and coding tasks.

GUIDELINES:
- Use function calling for tool operations. Do NOT output tool calls as text.
- Make focused, incremental code edits.
- Validate JSON tool arguments against schemas.`,
  postamble: `Think step by step for complex reasoning tasks.`,
  supportsThinking: false,
  toolFormat: "openai",
  recommendedMaxTokens: 8192,
  contextWindow: 128000,
};

const DEFAULT_TEMPLATE: SystemPromptTemplate = {
  family: "default",
  preamble: `You are a helpful AI assistant.

CRITICAL RULES:
- Use the function-calling API for all tool invocations.
- Never print tool calls as JSON text or wrap them in markdown.
- Make focused, incremental changes when editing files.
- Ensure all JSON arguments are valid and match tool schemas exactly.`,
  postamble: ``,
  supportsThinking: false,
  toolFormat: "openai",
  recommendedMaxTokens: 8192,
  contextWindow: 128000,
};

function normalizeModelName(name: string): string {
  return name.toLowerCase().replace(/[-_\.]/g, "");
}

function detectFamily(modelName: string): string {
  const n = normalizeModelName(modelName);
  if (n.includes("deepseek")) return "deepseek";
  if (n.includes("qwen")) return "qwen";
  if (n.includes("kimi")) return "kimi";
  if (n.includes("llama") || n.includes("nemotron")) return "llama";
  if (n.includes("glm")) return "glm";
  if (n.includes("nemotron")) return "nemotron";
  return "default";
}

const FAMILY_MAP: Record<string, SystemPromptTemplate> = {
  deepseek: DEEPSEEK_TEMPLATE,
  qwen: QWEN_TEMPLATE,
  kimi: KIMI_TEMPLATE,
  llama: LLAMA_TEMPLATE,
  glm: GLM_TEMPLATE,
  nemotron: NEMOTRON_TEMPLATE,
  default: DEFAULT_TEMPLATE,
};

export function getModelTemplate(modelName: string): SystemPromptTemplate {
  const family = detectFamily(modelName);
  return FAMILY_MAP[family] ?? DEFAULT_TEMPLATE;
}

export function buildSystemPrompt(
  baseSystem: string | null,
  modelName: string,
  options: {
    hasTools?: boolean;
    enableThinking?: boolean;
    toolUseAddendum?: string;
    retryHint?: string;
  } = {},
): string {
  const tpl = getModelTemplate(modelName);
  const parts: string[] = [];

  if (tpl.preamble) parts.push(tpl.preamble);
  if (baseSystem) parts.push(baseSystem);
  if (options.toolUseAddendum && options.hasTools) parts.push(options.toolUseAddendum);
  if (options.retryHint) parts.push(options.retryHint);
  if (tpl.postamble) parts.push(tpl.postamble);
  if (options.enableThinking && tpl.supportsThinking && tpl.thinkingActivation) {
    parts.push(tpl.thinkingActivation);
  }

  return parts.join("\n\n");
}

export function getRecommendedMaxTokens(modelName: string): number {
  return getModelTemplate(modelName).recommendedMaxTokens;
}

export function getContextWindow(modelName: string): number {
  return getModelTemplate(modelName).contextWindow;
}
