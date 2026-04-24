/**
 * Request augmentation / code-aware file context injection.
 *
 * Three layers of intelligence on top of plain `#file:foo` substitution:
 *
 *   1. Explicit file references (#file, @file, `path/to/file.ext`)
 *      are resolved against the project dir and injected verbatim.
 *
 *   2. Symbol references in the user's prompt (`functionName()`,
 *      `class ClassName`, `import { foo } from "./bar"`) trigger a
 *      regex-based grep across the project to find files where those
 *      symbols are defined or used; the top matches are injected.
 *
 *   3. For each explicit file, we walk shallow imports (one hop) to add
 *      the immediate dependencies — gives the model the surrounding API
 *      surface without dumping the whole repo.
 *
 * The augmenter is bounded: total injected context never exceeds a
 * configurable token budget (default 12k tokens worth of file content).
 */

import { readFileSync, existsSync, statSync, readdirSync } from "node:fs";
import path from "node:path";

interface FileReference {
  raw: string;
  filePath: string;
  lineRange?: { start: number; end: number };
  /** "explicit" | "symbol" | "dep" */
  origin: "explicit" | "symbol" | "dep";
}

const FILE_REF_PATTERNS: RegExp[] = [
  /#file:\s*([^\s\n)]+)/g,
  /@file:\s*([^\s\n)]+)/g,
  /#file\s+([^\s\n)]+)/g,
  /@file\s+([^\s\n)]+)/g,
  /`([^`\n]+\.(?:ts|tsx|js|jsx|py|rs|go|java|cpp|c|h|md|json|yaml|yml|toml|css|scss|html|sh|sql))`/g,
];

const SOURCE_EXTS = new Set([
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".mjs",
  ".cjs",
  ".py",
  ".rs",
  ".go",
  ".java",
]);

const IGNORE_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  ".next",
  ".turbo",
  ".cache",
  "coverage",
  "__pycache__",
  "target",
  ".venv",
  "venv",
]);

const TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4;

interface AugmentationOptions {
  projectDir?: string;
  /** Inject project tree of depth N at the top of the addendum. */
  injectDirectoryTree?: boolean;
  /** Max bytes per file. */
  maxFileSize?: number;
  /** Total chars across all injected files. */
  maxTotalChars?: number;
  /** Walk imports for explicit files (one hop). Default true. */
  walkImports?: boolean;
  /** Symbol-based discovery from prompt text. Default true. */
  discoverBySymbol?: boolean;
}

interface AugmentationResult {
  systemAddendum: string;
  modifiedMessages: Array<{ role: string; content: string | unknown }>;
  injectedFiles: string[];
  details: {
    explicit: number;
    bySymbol: number;
    byImport: number;
    droppedOversize: number;
    droppedOverBudget: number;
  };
}

function extractFileReferences(text: string): FileReference[] {
  const refs: FileReference[] = [];
  const seen = new Set<string>();

  for (const pattern of FILE_REF_PATTERNS) {
    const matches = text.matchAll(pattern);
    for (const match of matches) {
      const raw = match[0]!;
      let filePath = match[1]!.trim();

      let lineRange: { start: number; end: number } | undefined;
      const lineMatch = filePath.match(/^(.+):(\d+)(?:-(\d+))?$/);
      if (lineMatch) {
        filePath = lineMatch[1]!;
        const start = parseInt(lineMatch[2]!, 10);
        const end = lineMatch[3] ? parseInt(lineMatch[3], 10) : start;
        lineRange = { start, end };
      }

      const key = `${filePath}:${lineRange?.start ?? 0}-${lineRange?.end ?? 0}`;
      if (seen.has(key)) continue;
      seen.add(key);
      refs.push({ raw, filePath, lineRange, origin: "explicit" });
    }
  }

  return refs;
}

function resolveFilePath(filePath: string, projectDir: string): string | null {
  const candidates = [
    path.resolve(projectDir, filePath),
    path.resolve(projectDir, "src", filePath),
    path.resolve(projectDir, "lib", filePath),
    path.resolve(projectDir, "apps", filePath),
    path.resolve(projectDir, "packages", filePath),
  ];
  for (const candidate of candidates) {
    try {
      if (existsSync(candidate) && statSync(candidate).isFile()) return candidate;
    } catch {
      /* ignore */
    }
  }
  return null;
}

function readFileSlice(
  filePath: string,
  lineRange?: { start: number; end: number },
): string | null {
  try {
    const content = readFileSync(filePath, "utf8");
    if (!lineRange) return content;
    const lines = content.split("\n");
    const start = Math.max(0, lineRange.start - 1);
    const end = Math.min(lines.length, lineRange.end);
    return lines.slice(start, end).join("\n");
  } catch {
    return null;
  }
}

function buildFileContextSnippet(ref: FileReference, content: string): string {
  let snippet = `<file path="${ref.filePath}" origin="${ref.origin}">\n`;
  if (ref.lineRange) {
    snippet += `<!-- lines ${ref.lineRange.start}-${ref.lineRange.end} -->\n`;
  }
  snippet += "\n" + content + "\n</file>\n";
  return snippet;
}

/**
 * Walk a directory up to maxDepth, returning relative file paths.
 */
function walkProject(dir: string, maxDepth: number, maxFiles: number): string[] {
  const out: string[] = [];

  function recurse(current: string, depth: number, rel: string): void {
    if (out.length >= maxFiles || depth > maxDepth) return;
    let names: string[];
    try {
      names = readdirSync(current);
    } catch {
      return;
    }
    for (const name of names) {
      if (out.length >= maxFiles) return;
      if (name.startsWith(".") && name !== ".github") continue;
      if (IGNORE_DIRS.has(name)) continue;
      const childRel = rel ? path.posix.join(rel, name) : name;
      const fullPath = path.join(current, name);
      let stat: ReturnType<typeof statSync>;
      try {
        stat = statSync(fullPath);
      } catch {
        continue;
      }
      if (stat.isDirectory()) {
        recurse(fullPath, depth + 1, childRel);
      } else if (stat.isFile()) {
        out.push(childRel);
      }
    }
  }

  recurse(dir, 0, "");
  return out;
}

function getDirectoryTree(dir: string, maxDepth: number): string {
  const files = walkProject(dir, maxDepth, 500);
  // Group by top-level dir for readability.
  files.sort();
  return files.slice(0, 200).join("\n");
}

/**
 * Heuristic symbol extractor: pulls identifiers that look like
 * function/class/component names from the user prompt.
 */
function extractSymbols(text: string): string[] {
  const out = new Set<string>();
  // CamelCase or PascalCase identifiers >= 4 chars.
  const camelRe = /\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b/g;
  let m: RegExpExecArray | null;
  while ((m = camelRe.exec(text)) !== null) out.add(m[1]!);

  // function-style call references: `funcName(`
  const callRe = /\b([a-z_][a-zA-Z0-9_]{3,})\s*\(/g;
  while ((m = callRe.exec(text)) !== null) {
    const name = m[1]!;
    // Skip extremely common noise.
    if (name === "console" || name === "function" || name === "return") continue;
    out.add(name);
  }

  // Quoted import-like paths: `from "./foo"` or `import("./bar")`
  const impRe = /(?:from|import)\s*\(?\s*["']([^"']+)["']/g;
  while ((m = impRe.exec(text)) !== null) out.add(m[1]!);

  return [...out].slice(0, 25);
}

/**
 * Find files in the project that mention a symbol. Returns relative paths.
 */
function findFilesWithSymbol(
  symbol: string,
  projectDir: string,
  maxResults: number,
): string[] {
  // Pre-compile a word-boundary regex.
  const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`\\b${escaped}\\b`);
  const candidates = walkProject(projectDir, 6, 2000);
  const matches: string[] = [];
  for (const rel of candidates) {
    const ext = path.extname(rel);
    if (!SOURCE_EXTS.has(ext)) continue;
    try {
      const abs = path.join(projectDir, rel);
      const stats = statSync(abs);
      if (stats.size > 250_000) continue;
      const content = readFileSync(abs, "utf8");
      if (re.test(content)) {
        matches.push(rel);
        if (matches.length >= maxResults) break;
      }
    } catch {
      /* ignore */
    }
  }
  return matches;
}

/**
 * Extract import targets from a TS/JS file. Returns raw spec strings.
 */
function extractImports(content: string): string[] {
  const out = new Set<string>();
  const importRe = /(?:import\s+(?:[^"'\n]+\s+from\s+)?|require\s*\(\s*)["']([^"']+)["']/g;
  let m: RegExpExecArray | null;
  while ((m = importRe.exec(content)) !== null) {
    const spec = m[1]!;
    if (spec.startsWith(".")) out.add(spec);
  }
  return [...out];
}

/**
 * Resolve a relative import spec against the source file's directory.
 */
function resolveImport(spec: string, sourceFile: string): string | null {
  const baseDir = path.dirname(sourceFile);
  const candidates = [spec, `${spec}.ts`, `${spec}.tsx`, `${spec}.js`, `${spec}.jsx`,
    path.join(spec, "index.ts"), path.join(spec, "index.tsx"),
    path.join(spec, "index.js")];
  for (const c of candidates) {
    const abs = path.resolve(baseDir, c);
    try {
      if (existsSync(abs) && statSync(abs).isFile()) return abs;
    } catch {
      /* ignore */
    }
  }
  return null;
}

export function augmentRequestWithFileContext(
  messages: Array<{ role: string; content: string | unknown }>,
  _systemPrompt: string | null,
  options: AugmentationOptions = {},
): AugmentationResult {
  const projectDir = options.projectDir ?? process.env["NIM_PROJECT_DIR"] ?? process.cwd();
  const maxFileSize = options.maxFileSize ?? 50_000;
  const maxTotalChars = options.maxTotalChars ?? 48_000; // ~12k tokens
  const walkImports = options.walkImports ?? true;
  const discoverBySymbol = options.discoverBySymbol ?? true;

  const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
  if (!lastUserMsg || typeof lastUserMsg.content !== "string") {
    return {
      systemAddendum: "",
      modifiedMessages: messages,
      injectedFiles: [],
      details: { explicit: 0, bySymbol: 0, byImport: 0, droppedOversize: 0, droppedOverBudget: 0 },
    };
  }

  const promptText = lastUserMsg.content;
  const explicit = extractFileReferences(promptText);

  const allRefs: FileReference[] = [...explicit];
  let bySymbolCount = 0;
  let byImportCount = 0;

  // Layer 2: discover via symbols.
  if (discoverBySymbol && explicit.length === 0) {
    const symbols = extractSymbols(promptText);
    const seenPaths = new Set(allRefs.map((r) => r.filePath));
    let symbolBudget = 5;
    for (const sym of symbols) {
      if (symbolBudget <= 0) break;
      const matches = findFilesWithSymbol(sym, projectDir, 2);
      for (const rel of matches) {
        if (seenPaths.has(rel)) continue;
        seenPaths.add(rel);
        allRefs.push({ raw: "", filePath: rel, origin: "symbol" });
        bySymbolCount++;
        symbolBudget--;
        if (symbolBudget <= 0) break;
      }
    }
  }

  // Layer 3: walk one hop of imports for explicit files.
  if (walkImports && explicit.length > 0) {
    const seenPaths = new Set(allRefs.map((r) => r.filePath));
    for (const ref of [...explicit]) {
      const resolved = resolveFilePath(ref.filePath, projectDir);
      if (!resolved) continue;
      const ext = path.extname(resolved);
      if (!SOURCE_EXTS.has(ext)) continue;
      let content: string;
      try {
        content = readFileSync(resolved, "utf8");
      } catch {
        continue;
      }
      const specs = extractImports(content);
      let depBudget = 4;
      for (const spec of specs) {
        if (depBudget <= 0) break;
        const depAbs = resolveImport(spec, resolved);
        if (!depAbs) continue;
        const rel = path.relative(projectDir, depAbs);
        if (seenPaths.has(rel)) continue;
        seenPaths.add(rel);
        allRefs.push({ raw: "", filePath: rel, origin: "dep" });
        byImportCount++;
        depBudget--;
      }
    }
  }

  if (allRefs.length === 0 && !options.injectDirectoryTree) {
    return {
      systemAddendum: "",
      modifiedMessages: messages,
      injectedFiles: [],
      details: { explicit: 0, bySymbol: 0, byImport: 0, droppedOversize: 0, droppedOverBudget: 0 },
    };
  }

  const injectedFiles: string[] = [];
  const snippets: string[] = [];
  let totalChars = 0;
  let droppedOversize = 0;
  let droppedOverBudget = 0;

  for (const ref of allRefs) {
    const resolved = resolveFilePath(ref.filePath, projectDir);
    if (!resolved) {
      snippets.push(`<!-- File not found: ${ref.filePath} -->`);
      continue;
    }
    let stats;
    try {
      stats = statSync(resolved);
    } catch {
      continue;
    }
    if (stats.size > maxFileSize) {
      snippets.push(
        `<!-- File too large to inline: ${ref.filePath} (${Math.round(stats.size / 1024)}KB) -->`,
      );
      droppedOversize++;
      continue;
    }
    const content = readFileSlice(resolved, ref.lineRange);
    if (content === null) continue;
    if (totalChars + content.length > maxTotalChars) {
      droppedOverBudget++;
      continue;
    }
    snippets.push(buildFileContextSnippet(ref, content));
    injectedFiles.push(ref.filePath);
    totalChars += content.length;
  }

  if (options.injectDirectoryTree) {
    const tree = getDirectoryTree(projectDir, 3);
    if (tree) snippets.unshift(`<project_structure>\n${tree}\n</project_structure>`);
  }

  if (snippets.length === 0) {
    return {
      systemAddendum: "",
      modifiedMessages: messages,
      injectedFiles: [],
      details: {
        explicit: explicit.length,
        bySymbol: bySymbolCount,
        byImport: byImportCount,
        droppedOversize,
        droppedOverBudget,
      },
    };
  }

  const tokenEstimate = Math.ceil(totalChars / TOKEN_ESTIMATE_CHARS_PER_TOKEN);
  const addendum =
    `## Project Context\n` +
    `(${injectedFiles.length} file${injectedFiles.length === 1 ? "" : "s"} injected, ~${tokenEstimate} tokens; ` +
    `${explicit.length} explicit, ${bySymbolCount} via symbol, ${byImportCount} via import)\n\n` +
    snippets.join("\n");

  // Strip explicit file refs from the user's message so it reads naturally.
  let modifiedUserContent = promptText;
  for (const ref of explicit) {
    if (ref.raw) modifiedUserContent = modifiedUserContent.replace(ref.raw, "").trim();
  }

  const modifiedMessages = messages.map((m) => {
    if (m === lastUserMsg) {
      return {
        ...m,
        content: modifiedUserContent || "Please help with the files in the project context above.",
      };
    }
    return m;
  });

  return {
    systemAddendum: addendum,
    modifiedMessages,
    injectedFiles,
    details: {
      explicit: explicit.length,
      bySymbol: bySymbolCount,
      byImport: byImportCount,
      droppedOversize,
      droppedOverBudget,
    },
  };
}

export function hasFileReferences(text: string): boolean {
  return extractFileReferences(text).length > 0;
}
