/**
 * Request augmentation / file context injection.
 *
 * Parses file references (#file, @file) in user messages and injects
 * relevant file contents into the system prompt. Also supports
 * directory tree injection for coding tasks.
 */

import { readFileSync, existsSync, statSync, readdirSync } from "node:fs";
import path from "node:path";

interface FileReference {
  raw: string;
  filePath: string;
  lineRange?: { start: number; end: number };
}

const FILE_REF_PATTERNS = [
  /#file:\s*([^\s\n]+)/g,
  /@file:\s*([^\s\n]+)/g,
  /#file\s+([^\s\n]+)/g,
  /@file\s+([^\s\n]+)/g,
  /`([^`]+\.(?:ts|tsx|js|jsx|py|rs|go|java|cpp|c|h|md|json|yaml|yml|toml|css|scss|html))`/g,
];

function extractFileReferences(text: string): FileReference[] {
  const refs: FileReference[] = [];
  const seen = new Set<string>();

  for (const pattern of FILE_REF_PATTERNS) {
    const matches = text.matchAll(pattern);
    for (const match of matches) {
      const raw = match[0]!;
      let filePath = match[1]!.trim();

      // Parse line ranges like src/foo.ts:10-20
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
      refs.push({ raw, filePath, lineRange });
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
  ];

  for (const candidate of candidates) {
    if (existsSync(candidate) && statSync(candidate).isFile()) {
      return candidate;
    }
  }
  return null;
}

function readFileSlice(filePath: string, lineRange?: { start: number; end: number }): string | null {
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

function getDirectoryTree(dir: string, prefix = "", maxDepth = 3, currentDepth = 0): string {
  if (currentDepth >= maxDepth) return "";

  try {
    const entries = readdirSync(dir, { withFileTypes: true });
    const lines: string[] = [];

    for (const entry of entries) {
      if (entry.name.startsWith(".") && entry.name !== ".github") continue;
      if (entry.name === "node_modules" || entry.name === "dist" || entry.name === ".git") continue;

      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        lines.push(`${prefix}${entry.name}/`);
        lines.push(getDirectoryTree(fullPath, prefix + "  ", maxDepth, currentDepth + 1));
      } else {
        lines.push(`${prefix}${entry.name}`);
      }
    }

    return lines.join("\n");
  } catch {
    return "";
  }
}

function buildFileContextSnippet(ref: FileReference, content: string): string {
  const lang = path.extname(ref.filePath).slice(1) || "text";
  let snippet = `<file path="${ref.filePath}">\n`;
  if (ref.lineRange) {
    snippet += `<!-- lines ${ref.lineRange.start}-${ref.lineRange.end} -->\n`;
  }
  snippet += "\n";
  snippet += content;
  snippet += "\n</file>\n";
  return snippet;
}

interface AugmentationResult {
  systemAddendum: string;
  modifiedMessages: Array<{ role: string; content: string | unknown }>;
  injectedFiles: string[];
}

export function augmentRequestWithFileContext(
  messages: Array<{ role: string; content: string | unknown }>,
  systemPrompt: string | null,
  options: {
    projectDir?: string;
    injectDirectoryTree?: boolean;
    maxFileSize?: number;
  } = {},
): AugmentationResult {
  const projectDir = options.projectDir ?? process.env["NIM_PROJECT_DIR"] ?? process.cwd();
  const maxFileSize = options.maxFileSize ?? 50000; // 50KB max per file

  const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
  if (!lastUserMsg || typeof lastUserMsg.content !== "string") {
    return { systemAddendum: "", modifiedMessages: messages, injectedFiles: [] };
  }

  const refs = extractFileReferences(lastUserMsg.content);
  if (refs.length === 0 && !options.injectDirectoryTree) {
    return { systemAddendum: "", modifiedMessages: messages, injectedFiles: [] };
  }

  const injectedFiles: string[] = [];
  const snippets: string[] = [];

  for (const ref of refs) {
    const resolved = resolveFilePath(ref.filePath, projectDir);
    if (!resolved) {
      snippets.push(`<!-- File not found: ${ref.filePath} -->`);
      continue;
    }

    try {
      const stats = statSync(resolved);
      if (stats.size > maxFileSize) {
        snippets.push(`<!-- File too large: ${ref.filePath} (${Math.round(stats.size / 1024)}KB) -->`);
        continue;
      }

      const content = readFileSlice(resolved, ref.lineRange);
      if (content !== null) {
        snippets.push(buildFileContextSnippet(ref, content));
        injectedFiles.push(ref.filePath);
      }
    } catch {
      snippets.push(`<!-- Error reading: ${ref.filePath} -->`);
    }
  }

  if (options.injectDirectoryTree) {
    const tree = getDirectoryTree(projectDir);
    if (tree) {
      snippets.unshift(`<project_structure>\n${tree}\n</project_structure>`);
    }
  }

  if (snippets.length === 0) {
    return { systemAddendum: "", modifiedMessages: messages, injectedFiles: [] };
  }

  const addendum = `## Context\n\nThe following files are relevant to the current request:\n\n${snippets.join("\n")}`;

  // Remove file references from the user message to avoid confusion
  let modifiedUserContent = lastUserMsg.content;
  for (const ref of refs) {
    modifiedUserContent = modifiedUserContent.replace(ref.raw, "").trim();
  }

  const modifiedMessages = messages.map((m) => {
    if (m === lastUserMsg) {
      return { ...m, content: modifiedUserContent || "Please help with the files mentioned above." };
    }
    return m;
  });

  return {
    systemAddendum: addendum,
    modifiedMessages,
    injectedFiles,
  };
}

/**
 * Check if a message contains file references that should trigger augmentation.
 */
export function hasFileReferences(text: string): boolean {
  return extractFileReferences(text).length > 0;
}
