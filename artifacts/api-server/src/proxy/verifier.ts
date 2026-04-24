/**
 * Verifier loop.
 *
 * After the assistant uses a tool to mutate code, run a configurable
 * verification step (typecheck, lint, test) over the touched files and
 * surface diagnostics back into the conversation as a `tool_result` so the
 * model can see and fix its own mistakes — the single largest accuracy
 * gain reported in coding-agent papers.
 *
 * The verifier is intentionally *external* to the model: it runs whatever
 * commands the user has configured for their project and returns
 * structured diagnostics. It does NOT call the model itself; the wiring in
 * `messages.ts` decides whether/how to feed the diagnostics back.
 */

import { spawn } from "node:child_process";
import path from "node:path";
import { existsSync } from "node:fs";

export interface VerifierDiagnostic {
  /** Tool that produced this diagnostic ("typecheck" | "lint" | "test"). */
  source: string;
  /** Severity hint extracted from the tool output. */
  severity: "error" | "warning" | "info";
  /** Project-relative file path. */
  file: string | null;
  line: number | null;
  column: number | null;
  message: string;
}

export interface VerifierResult {
  ran: Array<"typecheck" | "lint" | "test">;
  diagnostics: VerifierDiagnostic[];
  exitCode: number;
  durationMs: number;
  truncated: boolean;
}

interface RunOptions {
  cwd: string;
  /** Files (project-relative) the assistant just touched. Used to scope output. */
  touchedFiles?: string[];
  /** Total time budget across all verifier steps, in ms. */
  timeBudgetMs?: number;
  /** Override the typecheck command (default: env NIM_VERIFY_TYPECHECK_CMD). */
  typecheckCmd?: string | null;
  /** Override the lint command (default: env NIM_VERIFY_LINT_CMD). */
  lintCmd?: string | null;
  /** Override the test command (default: env NIM_VERIFY_TEST_CMD). Often unset. */
  testCmd?: string | null;
}

const DEFAULT_TIME_BUDGET = 20_000;
const MAX_DIAGNOSTICS = 50;
const MAX_OUTPUT_BYTES = 256 * 1024;

interface ProcResult {
  stdout: string;
  stderr: string;
  code: number;
  timedOut: boolean;
  durationMs: number;
}

function runCmd(cmd: string, cwd: string, timeoutMs: number): Promise<ProcResult> {
  return new Promise((resolve) => {
    const start = Date.now();
    const child = spawn(cmd, {
      cwd,
      shell: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, CI: "1", FORCE_COLOR: "0", NO_COLOR: "1" },
    });

    let stdout = "";
    let stderr = "";
    let outBytes = 0;
    let errBytes = 0;

    child.stdout?.on("data", (chunk: Buffer) => {
      if (outBytes < MAX_OUTPUT_BYTES) {
        const room = MAX_OUTPUT_BYTES - outBytes;
        const slice = chunk.length > room ? chunk.subarray(0, room) : chunk;
        stdout += slice.toString("utf8");
        outBytes += slice.length;
      }
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      if (errBytes < MAX_OUTPUT_BYTES) {
        const room = MAX_OUTPUT_BYTES - errBytes;
        const slice = chunk.length > room ? chunk.subarray(0, room) : chunk;
        stderr += slice.toString("utf8");
        errBytes += slice.length;
      }
    });

    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      try {
        child.kill("SIGTERM");
        setTimeout(() => {
          if (!child.killed) child.kill("SIGKILL");
        }, 1000).unref();
      } catch {
        /* ignore */
      }
    }, timeoutMs);
    timer.unref();

    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({
        stdout,
        stderr,
        code: code ?? -1,
        timedOut,
        durationMs: Date.now() - start,
      });
    });

    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({
        stdout: "",
        stderr: String(err),
        code: -1,
        timedOut: false,
        durationMs: Date.now() - start,
      });
    });
  });
}

/**
 * Parse `tsc --noEmit` style output:
 *   src/foo.ts(10,3): error TS2304: Cannot find name 'x'.
 */
function parseTscOutput(text: string): VerifierDiagnostic[] {
  const out: VerifierDiagnostic[] = [];
  const re = /^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+([A-Z]+\d+):\s+(.+)$/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    out.push({
      source: "typecheck",
      severity: m[4] === "error" ? "error" : "warning",
      file: m[1] ?? null,
      line: m[2] ? parseInt(m[2], 10) : null,
      column: m[3] ? parseInt(m[3], 10) : null,
      message: `${m[5]}: ${m[6]}`,
    });
    if (out.length >= MAX_DIAGNOSTICS) break;
  }
  return out;
}

/**
 * Parse ESLint --format=compact:
 *   /path/to/file.ts: line 1, col 1, Error - foo (rule)
 * and the older "stylish" output:
 *   /path/to/file.ts
 *     1:5  error  foo  rule
 */
function parseEslintOutput(text: string): VerifierDiagnostic[] {
  const out: VerifierDiagnostic[] = [];

  // compact format
  const compactRe =
    /^(.+?):\s+line\s+(\d+),\s+col\s+(\d+),\s+(Error|Warning|Info)\s+-\s+(.+?)(?:\s+\((.+?)\))?\s*$/gim;
  let m: RegExpExecArray | null;
  while ((m = compactRe.exec(text)) !== null) {
    out.push({
      source: "lint",
      severity: (m[4] ?? "").toLowerCase() === "error" ? "error" : "warning",
      file: m[1] ?? null,
      line: m[2] ? parseInt(m[2], 10) : null,
      column: m[3] ? parseInt(m[3], 10) : null,
      message: m[6] ? `${m[5]} (${m[6]})` : (m[5] ?? ""),
    });
    if (out.length >= MAX_DIAGNOSTICS) break;
  }
  if (out.length > 0) return out;

  // stylish format
  let currentFile: string | null = null;
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line.trim()) {
      currentFile = null;
      continue;
    }
    if (/^[\/.][^:\s]/.test(line) || /^[A-Za-z]:[\\\/]/.test(line)) {
      currentFile = line.trim();
      continue;
    }
    const sm = line.match(/^\s*(\d+):(\d+)\s+(error|warning|info)\s+(.+?)(?:\s+([\w-]+\/[\w-]+|[\w-]+))?\s*$/);
    if (sm && currentFile) {
      out.push({
        source: "lint",
        severity: sm[3] === "error" ? "error" : "warning",
        file: currentFile,
        line: parseInt(sm[1] ?? "0", 10),
        column: parseInt(sm[2] ?? "0", 10),
        message: sm[5] ? `${sm[4]} (${sm[5]})` : (sm[4] ?? ""),
      });
      if (out.length >= MAX_DIAGNOSTICS) break;
    }
  }
  return out;
}

/**
 * Generic test-output parser: extract lines that look like failures.
 * We don't try to be clever about every framework; we extract assertion
 * failures and FAIL lines, capping at MAX_DIAGNOSTICS.
 */
function parseTestOutput(text: string): VerifierDiagnostic[] {
  const out: VerifierDiagnostic[] = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i] ?? "";
    if (
      /^\s*(FAIL|✗|×|Error:|AssertionError|Expected:|Received:)/.test(ln) ||
      /^\s*\d+\)\s/.test(ln) // "1) test name"
    ) {
      // Try to find a file ref nearby.
      const fileRef = ln.match(/(\S+\.(?:ts|tsx|js|jsx|py|rs|go)):(\d+)(?::(\d+))?/);
      out.push({
        source: "test",
        severity: "error",
        file: fileRef?.[1] ?? null,
        line: fileRef?.[2] ? parseInt(fileRef[2], 10) : null,
        column: fileRef?.[3] ? parseInt(fileRef[3], 10) : null,
        message: ln.trim().slice(0, 500),
      });
      if (out.length >= MAX_DIAGNOSTICS) break;
    }
  }
  return out;
}

function filterToTouched(
  diags: VerifierDiagnostic[],
  touched: string[] | undefined,
  cwd: string,
): VerifierDiagnostic[] {
  if (!touched || touched.length === 0) return diags;
  const norm = new Set(
    touched.map((f) => path.resolve(cwd, f)),
  );
  return diags.filter((d) => {
    if (!d.file) return true; // unattributed errors still surface
    const abs = path.resolve(cwd, d.file);
    return norm.has(abs);
  });
}

/**
 * Run the configured verifier steps. Each step is independent and bounded
 * by a share of the total time budget.
 */
export async function runVerifier(opts: RunOptions): Promise<VerifierResult> {
  if (!existsSync(opts.cwd)) {
    return {
      ran: [],
      diagnostics: [],
      exitCode: -1,
      durationMs: 0,
      truncated: false,
    };
  }

  const typecheckCmd =
    opts.typecheckCmd ?? process.env["NIM_VERIFY_TYPECHECK_CMD"] ?? null;
  const lintCmd = opts.lintCmd ?? process.env["NIM_VERIFY_LINT_CMD"] ?? null;
  const testCmd = opts.testCmd ?? process.env["NIM_VERIFY_TEST_CMD"] ?? null;

  const steps: Array<{ kind: "typecheck" | "lint" | "test"; cmd: string }> = [];
  if (typecheckCmd) steps.push({ kind: "typecheck", cmd: typecheckCmd });
  if (lintCmd) steps.push({ kind: "lint", cmd: lintCmd });
  if (testCmd) steps.push({ kind: "test", cmd: testCmd });

  if (steps.length === 0) {
    return {
      ran: [],
      diagnostics: [],
      exitCode: 0,
      durationMs: 0,
      truncated: false,
    };
  }

  const totalBudget = opts.timeBudgetMs ?? DEFAULT_TIME_BUDGET;
  const perStepBudget = Math.max(2_000, Math.floor(totalBudget / steps.length));

  const start = Date.now();
  const ran: VerifierResult["ran"] = [];
  const diagnostics: VerifierDiagnostic[] = [];
  let exitCode = 0;
  let truncated = false;

  for (const step of steps) {
    const remaining = totalBudget - (Date.now() - start);
    if (remaining <= 1_500) {
      truncated = true;
      break;
    }
    const budget = Math.min(perStepBudget, remaining);
    const r = await runCmd(step.cmd, opts.cwd, budget);
    ran.push(step.kind);

    const text = `${r.stdout}\n${r.stderr}`;
    let parsed: VerifierDiagnostic[] = [];
    if (step.kind === "typecheck") parsed = parseTscOutput(text);
    else if (step.kind === "lint") parsed = parseEslintOutput(text);
    else if (step.kind === "test") parsed = parseTestOutput(text);

    parsed = filterToTouched(parsed, opts.touchedFiles, opts.cwd);

    if (parsed.length === 0 && r.code !== 0) {
      // Tool failed but we couldn't parse structured output — surface a
      // single info diagnostic so the model still learns something went wrong.
      parsed.push({
        source: step.kind,
        severity: "error",
        file: null,
        line: null,
        column: null,
        message: `${step.kind} exited ${r.code}${r.timedOut ? " (timed out)" : ""}: ${text.split("\n").slice(-5).join(" ").slice(0, 400)}`,
      });
    }

    for (const d of parsed) {
      diagnostics.push(d);
      if (diagnostics.length >= MAX_DIAGNOSTICS) {
        truncated = true;
        break;
      }
    }
    if (r.code !== 0 && exitCode === 0) exitCode = r.code;
    if (diagnostics.length >= MAX_DIAGNOSTICS) break;
  }

  return {
    ran,
    diagnostics,
    exitCode,
    durationMs: Date.now() - start,
    truncated,
  };
}

/**
 * Format diagnostics as a tool_result-style block for the model.
 */
export function formatDiagnosticsForModel(result: VerifierResult): string {
  if (result.ran.length === 0) {
    return "Verifier: no checks configured (set NIM_VERIFY_TYPECHECK_CMD / NIM_VERIFY_LINT_CMD / NIM_VERIFY_TEST_CMD).";
  }
  if (result.diagnostics.length === 0 && result.exitCode === 0) {
    return `Verifier (${result.ran.join(", ")}): all checks passed in ${result.durationMs}ms.`;
  }
  const lines: string[] = [];
  lines.push(
    `Verifier (${result.ran.join(", ")}): ${result.diagnostics.length} diagnostic${result.diagnostics.length === 1 ? "" : "s"} in ${result.durationMs}ms`,
  );
  for (const d of result.diagnostics.slice(0, 30)) {
    const loc = d.file ? `${d.file}${d.line ? `:${d.line}` : ""}${d.column ? `:${d.column}` : ""}` : "(no file)";
    lines.push(`- [${d.source}/${d.severity}] ${loc} — ${d.message}`);
  }
  if (result.truncated) lines.push(`(truncated; ${result.diagnostics.length - 30} more not shown)`);
  return lines.join("\n");
}

/**
 * Last-result snapshot for the dashboard.
 */
let lastResult: { result: VerifierResult; at: number; cwd: string } | null = null;

export function recordVerifierRun(result: VerifierResult, cwd: string): void {
  lastResult = { result, at: Date.now(), cwd };
}

export function getLastVerifierRun(): { result: VerifierResult; at: number; cwd: string } | null {
  return lastResult;
}
