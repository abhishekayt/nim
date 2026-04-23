#!/usr/bin/env node
/**
 * nim-claude — wrap the Claude Code CLI with the NVIDIA NIM proxy.
 *
 * Default behavior: starts the local proxy server, then spawns `claude`
 * with ANTHROPIC_BASE_URL pointed at the proxy. All extra args are
 * forwarded to claude verbatim.
 *
 * Subcommands:
 *   keys list | add <key> [label] | remove <id> | enable <id> | disable <id>
 *   models list | add <name> | remove <id> | enable <id> | disable <id>
 *   status
 *   rotation key auto|manual | rotation model auto|manual
 *   use key <id>          # set active key
 *   use model <id>        # set active model
 *   proxy                 # start only the proxy (no claude)
 *   dashboard             # print dashboard URL
 *   help
 */
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import net from "node:net";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../../..");
const apiServerDir = path.resolve(repoRoot, "artifacts/api-server");
const apiServerEntry = path.resolve(apiServerDir, "dist/index.mjs");

const DEFAULT_PORT = Number(process.env["NIM_CLAUDE_PORT"] ?? 8787);

function logInfo(msg: string) { process.stderr.write(`\x1b[36m[nim-claude]\x1b[0m ${msg}\n`); }
function logErr(msg: string) { process.stderr.write(`\x1b[31m[nim-claude]\x1b[0m ${msg}\n`); }
function logOk(msg: string) { process.stderr.write(`\x1b[32m[nim-claude]\x1b[0m ${msg}\n`); }

async function isPortInUse(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const sock = net.createConnection({ port, host: "127.0.0.1" }, () => {
      sock.end(); resolve(true);
    });
    sock.on("error", () => resolve(false));
  });
}

async function waitForServer(port: number, timeoutMs = 15000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/api/healthz`);
      if (r.ok) return true;
    } catch { /* not ready */ }
    await new Promise((r) => setTimeout(r, 200));
  }
  return false;
}

async function ensureBuild(): Promise<void> {
  if (existsSync(apiServerEntry)) return;
  logInfo("Building proxy server (first run)…");
  await new Promise<void>((resolve, reject) => {
    const p = spawn("pnpm", ["--filter", "@workspace/api-server", "run", "build"], {
      cwd: repoRoot, stdio: "inherit",
    });
    p.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`build failed (code ${code})`))));
  });
}

async function startProxy(port: number): Promise<{ child: ChildProcess; baseUrl: string }> {
  if (await isPortInUse(port)) {
    const ok = await waitForServer(port, 1500);
    if (ok) {
      logInfo(`Reusing proxy already running on http://127.0.0.1:${port}`);
      return { child: null as unknown as ChildProcess, baseUrl: `http://127.0.0.1:${port}` };
    }
    throw new Error(`Port ${port} is in use by something else.`);
  }
  await ensureBuild();
  const child = spawn(process.execPath, ["--enable-source-maps", apiServerEntry], {
    cwd: apiServerDir,
    env: { ...process.env, PORT: String(port) },
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout?.on("data", (d) => process.stderr.write(`\x1b[90m[proxy]\x1b[0m ${d}`));
  child.stderr?.on("data", (d) => process.stderr.write(`\x1b[90m[proxy]\x1b[0m ${d}`));
  const ready = await waitForServer(port);
  if (!ready) {
    child.kill();
    throw new Error("Proxy server failed to start within 15s");
  }
  logOk(`Proxy ready at http://127.0.0.1:${port}`);
  return { child, baseUrl: `http://127.0.0.1:${port}` };
}

function findClaudeBinary(): string | null {
  // Allow override
  if (process.env["CLAUDE_BIN"] && existsSync(process.env["CLAUDE_BIN"]!)) return process.env["CLAUDE_BIN"]!;
  // Search PATH
  const pathDirs = (process.env["PATH"] ?? "").split(path.delimiter);
  const names = ["claude", "claude.cmd", "claude.exe"];
  for (const dir of pathDirs) {
    for (const n of names) {
      const candidate = path.join(dir, n);
      if (existsSync(candidate)) return candidate;
    }
  }
  return null;
}

async function api(baseUrl: string, p: string, init?: RequestInit) {
  const r = await fetch(`${baseUrl}${p}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`${r.status}: ${text}`);
  try { return JSON.parse(text); } catch { return text; }
}

function fmtBadge(item: { enabled: boolean; rateLimitedUntil?: number | null; failingUntil?: number | null; lastError?: string | null }) {
  if (!item.enabled) return "\x1b[90mdisabled\x1b[0m";
  const cd = item.rateLimitedUntil ?? item.failingUntil;
  if (cd && cd > Date.now()) {
    const s = Math.ceil((cd - Date.now()) / 1000);
    return `\x1b[33mcooldown ${s < 60 ? `${s}s` : `${Math.ceil(s / 60)}m`}\x1b[0m`;
  }
  if (item.lastError) return "\x1b[31merror\x1b[0m";
  return "\x1b[32mready\x1b[0m";
}

async function ensureProxyForCli(port: number): Promise<{ baseUrl: string; ownChild: ChildProcess | null }> {
  if (await isPortInUse(port) && (await waitForServer(port, 1500))) {
    return { baseUrl: `http://127.0.0.1:${port}`, ownChild: null };
  }
  const started = await startProxy(port);
  return { baseUrl: started.baseUrl, ownChild: started.child };
}

async function cmdStatus(port: number) {
  const { baseUrl, ownChild } = await ensureProxyForCli(port);
  try {
    const cfg = await api(baseUrl, "/admin/status");
    console.log(`\nProxy:    ${baseUrl}`);
    console.log(`NIM URL:  ${cfg.rotation.nimBaseUrl}`);
    console.log(`Key mode:    ${cfg.rotation.keyMode}    (active: ${cfg.rotation.activeKeyId ?? "—"})`);
    console.log(`Model mode:  ${cfg.rotation.modelMode}  (active: ${cfg.rotation.activeModelId ?? "—"})\n`);
    console.log("Keys:");
    if (cfg.keys.length === 0) console.log("  (none — add one with `nim-claude keys add <nvapi-…>`)");
    for (const k of cfg.keys) console.log(`  ${k.id}  ${k.label.padEnd(16)} ${k.masked.padEnd(14)} ${fmtBadge(k)}  ✓${k.successCount} ✗${k.errorCount}`);
    console.log("\nModels:");
    for (const m of cfg.models) console.log(`  ${m.id}  ${m.name.padEnd(48)} ${fmtBadge(m)}  ✓${m.successCount} ✗${m.errorCount}`);
    console.log();
  } finally {
    if (ownChild) ownChild.kill();
  }
}

async function cmdKeys(args: string[], port: number) {
  const sub = args[0];
  const { baseUrl, ownChild } = await ensureProxyForCli(port);
  try {
    if (!sub || sub === "list") return cmdStatus(port);
    if (sub === "add") {
      const key = args[1]; const label = args.slice(2).join(" ") || undefined;
      if (!key) throw new Error("Usage: nim-claude keys add <nvapi-key> [label]");
      await api(baseUrl, "/admin/keys", { method: "POST", body: JSON.stringify({ key, label }) });
      logOk("Key added"); return;
    }
    const id = args[1];
    if (!id) throw new Error(`Usage: nim-claude keys ${sub} <id>`);
    if (sub === "remove") { await api(baseUrl, `/admin/keys/${id}`, { method: "DELETE" }); logOk("Removed"); return; }
    if (sub === "enable") { await api(baseUrl, `/admin/keys/${id}`, { method: "PATCH", body: JSON.stringify({ enabled: true }) }); logOk("Enabled"); return; }
    if (sub === "disable") { await api(baseUrl, `/admin/keys/${id}`, { method: "PATCH", body: JSON.stringify({ enabled: false }) }); logOk("Disabled"); return; }
    if (sub === "clear") { await api(baseUrl, `/admin/keys/${id}/clear-cooldown`, { method: "POST" }); logOk("Cleared cooldown"); return; }
    throw new Error(`Unknown subcommand: keys ${sub}`);
  } finally { if (ownChild) ownChild.kill(); }
}

async function cmdModels(args: string[], port: number) {
  const sub = args[0];
  const { baseUrl, ownChild } = await ensureProxyForCli(port);
  try {
    if (!sub || sub === "list") return cmdStatus(port);
    if (sub === "add") {
      const name = args[1];
      if (!name) throw new Error("Usage: nim-claude models add <model-name>");
      await api(baseUrl, "/admin/models", { method: "POST", body: JSON.stringify({ name }) });
      logOk("Model added"); return;
    }
    const id = args[1];
    if (!id) throw new Error(`Usage: nim-claude models ${sub} <id>`);
    if (sub === "remove") { await api(baseUrl, `/admin/models/${id}`, { method: "DELETE" }); logOk("Removed"); return; }
    if (sub === "enable") { await api(baseUrl, `/admin/models/${id}`, { method: "PATCH", body: JSON.stringify({ enabled: true }) }); logOk("Enabled"); return; }
    if (sub === "disable") { await api(baseUrl, `/admin/models/${id}`, { method: "PATCH", body: JSON.stringify({ enabled: false }) }); logOk("Disabled"); return; }
    if (sub === "clear") { await api(baseUrl, `/admin/models/${id}/clear-cooldown`, { method: "POST" }); logOk("Cleared cooldown"); return; }
    throw new Error(`Unknown subcommand: models ${sub}`);
  } finally { if (ownChild) ownChild.kill(); }
}

async function cmdRotation(args: string[], port: number) {
  const target = args[0]; const mode = args[1];
  if (!["key", "model"].includes(target ?? "") || !["auto", "manual"].includes(mode ?? "")) {
    throw new Error("Usage: nim-claude rotation key|model auto|manual");
  }
  const { baseUrl, ownChild } = await ensureProxyForCli(port);
  try {
    const body = target === "key" ? { keyMode: mode } : { modelMode: mode };
    await api(baseUrl, "/admin/rotation", { method: "PUT", body: JSON.stringify(body) });
    logOk(`${target} mode set to ${mode}`);
  } finally { if (ownChild) ownChild.kill(); }
}

async function cmdUse(args: string[], port: number) {
  const target = args[0]; const id = args[1];
  if (!["key", "model"].includes(target ?? "") || !id) throw new Error("Usage: nim-claude use key|model <id>");
  const { baseUrl, ownChild } = await ensureProxyForCli(port);
  try {
    const body = target === "key" ? { activeKeyId: id } : { activeModelId: id };
    await api(baseUrl, "/admin/rotation", { method: "PUT", body: JSON.stringify(body) });
    logOk(`Active ${target} → ${id}`);
  } finally { if (ownChild) ownChild.kill(); }
}

async function runClaudeWrapped(claudeArgs: string[], port: number) {
  const { child: proxyChild, baseUrl } = await startProxy(port);

  const claudeBin = findClaudeBinary();
  if (!claudeBin) {
    logErr("Could not find `claude` on your PATH.");
    logErr("Install Claude Code first: https://docs.claude.com/en/docs/claude-code");
    logErr("Or set CLAUDE_BIN=/path/to/claude.");
    if (proxyChild) proxyChild.kill();
    process.exit(1);
  }

  // Quickly verify keys are configured
  try {
    const cfg = await api(baseUrl, "/admin/status");
    if (!cfg.keys || cfg.keys.length === 0) {
      logErr("No NVIDIA NIM keys configured.");
      logErr("Add one with:  nim-claude keys add nvapi-XXXXXXXX");
      logErr(`Or open the dashboard: ${baseUrl}`);
      if (proxyChild) proxyChild.kill();
      process.exit(1);
    }
  } catch { /* ignore — let claude try */ }

  logOk(`Launching claude → ${baseUrl}`);
  const env = {
    ...process.env,
    ANTHROPIC_BASE_URL: baseUrl,
    ANTHROPIC_API_KEY: process.env["ANTHROPIC_API_KEY"] || "nim-proxy-placeholder",
    // Some older Claude Code versions read this:
    ANTHROPIC_AUTH_TOKEN: process.env["ANTHROPIC_AUTH_TOKEN"] || "nim-proxy-placeholder",
  };

  const claude = spawn(claudeBin, claudeArgs, { env, stdio: "inherit" });

  const cleanup = () => {
    if (proxyChild && !proxyChild.killed) proxyChild.kill();
  };
  process.on("SIGINT", () => { claude.kill("SIGINT"); });
  process.on("SIGTERM", () => { claude.kill("SIGTERM"); });
  claude.on("exit", (code, signal) => {
    cleanup();
    if (signal) process.kill(process.pid, signal);
    else process.exit(code ?? 0);
  });
}

async function cmdProxyOnly(port: number) {
  const { child, baseUrl } = await startProxy(port);
  logOk(`Dashboard:  ${baseUrl}`);
  logOk(`Use as ANTHROPIC_BASE_URL=${baseUrl}`);
  if (!child) return; // already running externally
  const wait = new Promise<void>((resolve) => {
    process.on("SIGINT", () => { child.kill(); resolve(); });
    process.on("SIGTERM", () => { child.kill(); resolve(); });
    child.on("exit", () => resolve());
  });
  await wait;
}

function printHelp() {
  process.stdout.write(`nim-claude — Claude Code wrapped with the NVIDIA NIM proxy.

USAGE
  nim-claude [claude-args…]                Start proxy and launch Claude Code through it
  nim-claude proxy                         Run only the proxy (with dashboard)
  nim-claude dashboard                     Print the dashboard URL
  nim-claude status                        Show keys, models, and rotation state
  nim-claude keys list                     List configured keys
  nim-claude keys add <nvapi-…> [label]    Add a NIM key (max 3)
  nim-claude keys remove <id>              Remove a key
  nim-claude keys enable|disable <id>      Toggle a key
  nim-claude keys clear <id>               Clear a key's rate-limit cooldown
  nim-claude models list                   List configured models
  nim-claude models add <model-name>       Add a model
  nim-claude models remove <id>            Remove a model
  nim-claude models enable|disable <id>    Toggle a model
  nim-claude models clear <id>             Clear a model's failure cooldown
  nim-claude rotation key auto|manual      Set key rotation mode
  nim-claude rotation model auto|manual    Set model rotation mode
  nim-claude use key <id>                  Set the active key (used in manual mode, preferred in auto)
  nim-claude use model <id>                Set the active model
  nim-claude help                          Show this help

ENV VARS
  NIM_CLAUDE_PORT      Local proxy port (default 8787)
  NIM_API_KEY_1/2/3    Bootstrap up to 3 NIM keys on first run
  NIM_BASE_URL         NIM endpoint (default https://integrate.api.nvidia.com/v1)
  CLAUDE_BIN           Override path to the claude binary
`);
}

async function main() {
  const argv = process.argv.slice(2);
  const port = DEFAULT_PORT;
  const cmd = argv[0];

  try {
    if (!cmd) { await runClaudeWrapped([], port); return; }
    if (cmd === "help" || cmd === "--help" || cmd === "-h") { printHelp(); return; }
    if (cmd === "status") return await cmdStatus(port);
    if (cmd === "keys") return await cmdKeys(argv.slice(1), port);
    if (cmd === "models") return await cmdModels(argv.slice(1), port);
    if (cmd === "rotation") return await cmdRotation(argv.slice(1), port);
    if (cmd === "use") return await cmdUse(argv.slice(1), port);
    if (cmd === "proxy") return await cmdProxyOnly(port);
    if (cmd === "dashboard") { console.log(`http://127.0.0.1:${port}`); return; }
    // Anything else → forward to claude
    await runClaudeWrapped(argv, port);
  } catch (e) {
    logErr(e instanceof Error ? e.message : String(e));
    process.exit(1);
  }
}

main();
