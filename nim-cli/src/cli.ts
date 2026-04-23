/**
 * `nim` — wrap the Claude Code CLI with the NVIDIA NIM proxy.
 *
 * Default behavior: starts the in-process proxy server, then spawns `claude`
 * with ANTHROPIC_BASE_URL pointed at the proxy. All extra args are forwarded
 * to claude verbatim, giving you the full Claude Code experience.
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import net from "node:net";
import { startServer } from "./server";
import { loadConfig, updateConfig, publicConfig, configFilePath, PROVIDER_PRESETS } from "./proxy/store";
import { getTelemetry } from "./proxy/telemetry";
import { loadProjectConfig, projectConfigPath } from "./proxy/projectConfig";

const DEFAULT_PORT = Number(process.env["NIM_PORT"] ?? 8787);

const c = {
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  dim: (s: string) => `\x1b[90m${s}\x1b[0m`,
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
};

function info(msg: string) { process.stderr.write(`${c.cyan("[nim]")} ${msg}\n`); }
function ok(msg: string) { process.stderr.write(`${c.green("[nim]")} ${msg}\n`); }
function err(msg: string) { process.stderr.write(`${c.red("[nim]")} ${msg}\n`); }

async function isPortInUse(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const s = net.createConnection({ port, host: "127.0.0.1" }, () => { s.end(); resolve(true); });
    s.on("error", () => resolve(false));
  });
}

async function ensureLocalProxy(port: number): Promise<{ baseUrl: string; isOurs: boolean }> {
  const baseUrl = `http://127.0.0.1:${port}`;
  if (await isPortInUse(port)) {
    try {
      const r = await fetch(`${baseUrl}/api/healthz`);
      if (r.ok) return { baseUrl, isOurs: false };
    } catch { /* fall through */ }
    throw new Error(`Port ${port} is in use by something else. Set NIM_PORT to a free port.`);
  }
  await startServer(port);
  return { baseUrl, isOurs: true };
}

/**
 * Pre-seed `~/.claude.json` so Claude Code skips its onboarding wizard and
 * never asks the user to log in. Safe to call repeatedly — only fills in
 * missing flags, never overwrites existing user data.
 */
function bypassClaudeAuth(): void {
  try {
    const home = os.homedir();
    const cfgPath = path.join(home, ".claude.json");
    let cfg: Record<string, unknown> = {};
    if (existsSync(cfgPath)) {
      try { cfg = JSON.parse(readFileSync(cfgPath, "utf8")) as Record<string, unknown>; } catch { cfg = {}; }
    }
    const before = JSON.stringify(cfg);
    if (cfg["hasCompletedOnboarding"] === undefined) cfg["hasCompletedOnboarding"] = true;
    if (cfg["bypassPermissionsModeAccepted"] === undefined) cfg["bypassPermissionsModeAccepted"] = true;
    if (cfg["userID"] === undefined) cfg["userID"] = "nim-proxy-user";
    if (cfg["projects"] === undefined) cfg["projects"] = {};
    if (JSON.stringify(cfg) !== before) {
      mkdirSync(home, { recursive: true });
      writeFileSync(cfgPath, JSON.stringify(cfg, null, 2), "utf8");
    }
  } catch {
    // Non-fatal — Claude Code will still work, the user may just see the
    // onboarding prompt once.
  }
}

function findClaudeBinary(): string | null {
  if (process.env["CLAUDE_BIN"] && existsSync(process.env["CLAUDE_BIN"]!)) return process.env["CLAUDE_BIN"]!;
  const dirs = (process.env["PATH"] ?? "").split(path.delimiter);
  const names = process.platform === "win32" ? ["claude.cmd", "claude.exe", "claude"] : ["claude"];
  for (const dir of dirs) {
    for (const n of names) {
      const candidate = path.join(dir, n);
      if (existsSync(candidate)) return candidate;
    }
  }
  return null;
}

function fmtBadge(item: { enabled: boolean; rateLimitedUntil?: number | null; failingUntil?: number | null; lastError?: string | null }): string {
  if (!item.enabled) return c.dim("disabled");
  const cd = item.rateLimitedUntil ?? item.failingUntil;
  if (cd && cd > Date.now()) {
    const s = Math.ceil((cd - Date.now()) / 1000);
    return c.yellow(`cooldown ${s < 60 ? `${s}s` : `${Math.ceil(s / 60)}m`}`);
  }
  if (item.lastError) return c.red("error");
  return c.green("ready");
}

async function cmdStatus() {
  const cfg = publicConfig(await loadConfig());
  console.log();
  console.log(c.bold("Config file:") + "  " + configFilePath());
  console.log(c.bold("NIM URL:    ") + "  " + cfg.rotation.nimBaseUrl);
  console.log(c.bold("Key mode:   ") + "  " + cfg.rotation.keyMode + c.dim(`   active=${cfg.rotation.activeKeyId ?? "—"}`));
  console.log(c.bold("Model mode: ") + "  " + cfg.rotation.modelMode + c.dim(`   active=${cfg.rotation.activeModelId ?? "—"}`));
  console.log();
  console.log(c.bold("Keys"));
  if (cfg.keys.length === 0) console.log("  " + c.dim("(none — add one with `nim keys add <nvapi-…>`)"));
  for (const k of cfg.keys) {
    const provider = (k.providerId ?? "nim").padEnd(11);
    console.log(`  ${c.cyan(k.id.padEnd(4))} ${k.label.padEnd(16)} ${c.dim(k.masked.padEnd(14))} ${c.dim(provider)} ${fmtBadge(k)}  ${c.dim(`✓${k.successCount} ✗${k.errorCount}`)}`);
    if (k.lastError) console.log("    " + c.dim(k.lastError.slice(0, 100)));
  }
  const proj = loadProjectConfig();
  const projPath = projectConfigPath();
  if (projPath) {
    console.log();
    console.log(c.bold("Project config:") + " " + c.dim(projPath));
    console.log("  " + c.dim(JSON.stringify(proj)));
  }
  console.log();
  console.log(c.bold("Models") + c.dim("    (auto-routed by request type: coding / reasoning / vision / general)"));
  for (const m of cfg.models) {
    const cats = (m.categories ?? ["general"]).join(",");
    console.log(`  ${c.cyan(m.id.padEnd(4))} ${m.name.padEnd(46)} ${c.dim(cats.padEnd(20))} ${fmtBadge(m)}  ${c.dim(`✓${m.successCount} ✗${m.errorCount}`)}`);
    if (m.lastError) console.log("    " + c.dim(m.lastError.slice(0, 100)));
  }
  console.log();
}

async function cmdKeys(args: string[]) {
  const sub = args[0];
  if (!sub || sub === "list") return cmdStatus();
  if (sub === "add") {
    // Parse optional --provider <id> flag, anywhere in the arg list.
    const rest = args.slice(1);
    let providerId = "nim";
    const flagIdx = rest.findIndex((a) => a === "--provider" || a === "-p");
    if (flagIdx >= 0) {
      providerId = rest[flagIdx + 1] ?? "";
      if (!PROVIDER_PRESETS[providerId]) {
        throw new Error(`Unknown provider "${providerId}". Available: ${Object.keys(PROVIDER_PRESETS).join(", ")}`);
      }
      rest.splice(flagIdx, 2);
    }
    const key = rest[0]; const label = rest.slice(1).join(" ") || undefined;
    if (!key) throw new Error("Usage: nim keys add [--provider nim|groq|openrouter|together|deepinfra] <api-key> [label]");
    await updateConfig((c2) => {
      if (c2.keys.length >= 3) throw new Error("Maximum 3 keys allowed (remove one first).");
      const id = `k${Date.now().toString(36)}`;
      c2.keys.push({ id, label: label || `Key ${c2.keys.length + 1}`, key: key.trim(), enabled: true, rateLimitedUntil: null, lastError: null, successCount: 0, errorCount: 0, providerId });
      if (!c2.rotation.activeKeyId) c2.rotation.activeKeyId = id;
    });
    return ok(`Key added (provider: ${providerId}).`);
  }
  const id = args[1];
  if (!id) throw new Error(`Usage: nim keys ${sub} <id>`);
  if (sub === "remove") {
    await updateConfig((c2) => {
      c2.keys = c2.keys.filter((x) => x.id !== id);
      if (c2.rotation.activeKeyId === id) c2.rotation.activeKeyId = c2.keys[0]?.id ?? null;
    });
    return ok("Removed.");
  }
  if (sub === "enable" || sub === "disable") {
    await updateConfig((c2) => {
      const k = c2.keys.find((x) => x.id === id); if (!k) throw new Error("Key not found");
      k.enabled = sub === "enable";
    });
    return ok(`${sub === "enable" ? "Enabled" : "Disabled"}.`);
  }
  if (sub === "clear") {
    await updateConfig((c2) => {
      const k = c2.keys.find((x) => x.id === id); if (!k) throw new Error("Key not found");
      k.rateLimitedUntil = null; k.lastError = null;
    });
    return ok("Cleared cooldown.");
  }
  throw new Error(`Unknown subcommand: keys ${sub}`);
}

async function cmdModels(args: string[]) {
  const sub = args[0];
  if (!sub || sub === "list") return cmdStatus();
  if (sub === "add") {
    const name = args[1];
    if (!name) throw new Error("Usage: nim models add <model-name>");
    await updateConfig((c2) => {
      const id = `m${Date.now().toString(36)}`;
      c2.models.push({ id, name: name.trim(), enabled: true, failingUntil: null, lastError: null, successCount: 0, errorCount: 0 });
    });
    return ok("Model added.");
  }
  const id = args[1];
  if (!id) throw new Error(`Usage: nim models ${sub} <id>`);
  if (sub === "remove") {
    await updateConfig((c2) => {
      c2.models = c2.models.filter((x) => x.id !== id);
      if (c2.rotation.activeModelId === id) c2.rotation.activeModelId = c2.models[0]?.id ?? null;
    });
    return ok("Removed.");
  }
  if (sub === "enable" || sub === "disable") {
    await updateConfig((c2) => {
      const m = c2.models.find((x) => x.id === id); if (!m) throw new Error("Model not found");
      m.enabled = sub === "enable";
    });
    return ok(`${sub === "enable" ? "Enabled" : "Disabled"}.`);
  }
  if (sub === "clear") {
    await updateConfig((c2) => {
      const m = c2.models.find((x) => x.id === id); if (!m) throw new Error("Model not found");
      m.failingUntil = null; m.lastError = null;
    });
    return ok("Cleared cooldown.");
  }
  throw new Error(`Unknown subcommand: models ${sub}`);
}

async function cmdRotation(args: string[]) {
  const target = args[0]; const mode = args[1];
  if (!["key", "model"].includes(target ?? "") || !["auto", "manual"].includes(mode ?? "")) {
    throw new Error("Usage: nim rotation key|model auto|manual");
  }
  await updateConfig((c2) => {
    if (target === "key") c2.rotation.keyMode = mode as "auto" | "manual";
    else c2.rotation.modelMode = mode as "auto" | "manual";
  });
  ok(`${target} rotation set to ${mode}.`);
}

async function cmdUse(args: string[]) {
  const target = args[0]; const id = args[1];
  if (!["key", "model"].includes(target ?? "") || !id) throw new Error("Usage: nim use key|model <id>");
  await updateConfig((c2) => {
    if (target === "key") {
      if (!c2.keys.some((k) => k.id === id)) throw new Error("Key not found");
      c2.rotation.activeKeyId = id;
    } else {
      if (!c2.models.some((m) => m.id === id)) throw new Error("Model not found");
      c2.rotation.activeModelId = id;
    }
  });
  ok(`Active ${target} → ${id}`);
}

async function cmdTelemetry() {
  const { records, summary } = getTelemetry(20);
  console.log();
  console.log(c.bold("Telemetry") + c.dim(`  (last ${records.length} of ${summary.total} requests)`));
  console.log(`  total=${summary.total}  errors=${summary.errors}  cached=${summary.cached}  hit-rate=${(summary.cacheHitRate * 100).toFixed(0)}%  avg=${summary.avgLatencyMs}ms`);
  console.log();
  console.log(c.bold("Per-model"));
  if (summary.perModel.length === 0) console.log("  " + c.dim("no requests yet — start using nim and check back"));
  for (const m of summary.perModel) {
    console.log(`  ${m.modelName.padEnd(46)} ${c.dim(`reqs=${String(m.count).padStart(3)} err=${m.errors} cache=${m.cached} avg=${m.avgLatencyMs}ms p95=${m.p95LatencyMs}ms tok in/out=${m.totalInputTokens}/${m.totalOutputTokens}`)}`);
  }
  console.log();
  console.log(c.bold("Recent"));
  if (records.length === 0) console.log("  " + c.dim("no requests yet"));
  for (const r of records) {
    const t = new Date(r.ts).toISOString().slice(11, 19);
    const status = r.status === "ok" ? c.green("ok    ") : r.status === "cached" ? c.cyan("cached") : c.red("error ");
    const stream = r.streaming ? "S" : " ";
    console.log(`  ${c.dim(t)} ${status} ${stream} ${r.modelName.padEnd(46)} ${c.dim(`${r.latencyMs}ms ${r.categories.join(",") || "-"}`)}`);
    if (r.errorMessage) console.log("    " + c.dim(r.errorMessage.slice(0, 100)));
  }
  console.log();
}

async function cmdProxyOnly(port: number) {
  await ensureLocalProxy(port);
  ok(`Proxy ready at ${c.bold(`http://127.0.0.1:${port}`)}`);
  ok(`Dashboard:  ${c.bold(`http://127.0.0.1:${port}/`)}`);
  ok("Press Ctrl+C to stop.");
  await new Promise<void>((resolve) => {
    process.on("SIGINT", resolve);
    process.on("SIGTERM", resolve);
  });
}

async function runClaudeWrapped(claudeArgs: string[], port: number) {
  const claudeBin = findClaudeBinary();
  if (!claudeBin) {
    err("Could not find `claude` on your PATH.");
    err("Install Claude Code first:  npm install -g @anthropic-ai/claude-code");
    err("Or set CLAUDE_BIN=/path/to/claude.");
    process.exit(1);
  }

  const cfg = await loadConfig();
  if (cfg.keys.length === 0) {
    err("No NVIDIA NIM keys configured.");
    err("Add one with:  " + c.bold("nim keys add nvapi-XXXXXXXX"));
    process.exit(1);
  }

  const { baseUrl } = await ensureLocalProxy(port);
  ok(`Proxy ready at ${baseUrl}  ${c.dim(`(${cfg.keys.length} key${cfg.keys.length === 1 ? "" : "s"}, ${cfg.models.filter((m) => m.enabled).length} models)`)}`);
  bypassClaudeAuth();
  info(`Launching ${c.bold("claude")} → traffic routed through NIM`);

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    ANTHROPIC_BASE_URL: baseUrl,
    // Force a placeholder so Claude Code never tries to OAuth-login or read
    // the user's real Anthropic credentials. The proxy ignores these values.
    ANTHROPIC_API_KEY: "nim-proxy-placeholder",
    ANTHROPIC_AUTH_TOKEN: "nim-proxy-placeholder",
    // Skip first-run onboarding flow.
    CLAUDE_CODE_SKIP_ONBOARDING: "1",
    // Don't phone home for telemetry / non-essential traffic.
    DISABLE_TELEMETRY: "1",
    DISABLE_ERROR_REPORTING: "1",
    DISABLE_NON_ESSENTIAL_MODEL_CALLS: "1",
    DISABLE_AUTOUPDATER: "1",
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
  };

  const claude = spawn(claudeBin, claudeArgs, { env, stdio: "inherit",shell:true });

  process.on("SIGINT", () => claude.kill("SIGINT"));
  process.on("SIGTERM", () => claude.kill("SIGTERM"));
  claude.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    else process.exit(code ?? 0);
  });
}

function printHelp() {
  process.stdout.write(`${c.bold("nim")} — Claude Code wrapped with the NVIDIA NIM free API.

${c.bold("USAGE")}
  ${c.cyan("nim")} [claude-args…]                     Start proxy and launch Claude Code through it
  ${c.cyan("nim proxy")}                              Run only the proxy + dashboard
  ${c.cyan("nim status")}                             Show keys, models, and rotation state
  ${c.cyan("nim keys list")}
  ${c.cyan("nim keys add")} [--provider <id>] <key> [label]  Add an API key (max 3)
                                          providers: nim (default), groq, openrouter, together, deepinfra
  ${c.cyan("nim keys remove")} <id>
  ${c.cyan("nim keys enable|disable")} <id>
  ${c.cyan("nim keys clear")} <id>                    Clear a key's rate-limit cooldown
  ${c.cyan("nim models list")}
  ${c.cyan("nim models add")} <model-name>
  ${c.cyan("nim models remove")} <id>
  ${c.cyan("nim models enable|disable")} <id>
  ${c.cyan("nim models clear")} <id>
  ${c.cyan("nim rotation key|model auto|manual")}     Set rotation mode
  ${c.cyan("nim use key|model")} <id>                 Set the active key/model
  ${c.cyan("nim telemetry")}                          Show recent requests, latency, cache hits
  ${c.cyan("nim help")}                               Show this help

${c.bold("PER-PROJECT CONFIG")}
  Drop a ${c.cyan(".nimrc")} (JSON) in your project root to override defaults for that repo:
    {"category":"coding","model":"qwen/qwen2.5-coder-32b-instruct","cache":false}

${c.bold("ENV VARS")}
  NIM_PORT           Local proxy port (default 8787)
  NIM_API_KEY_1/2/3  Bootstrap up to 3 NIM keys on first run
  NIM_BASE_URL       NIM endpoint (default https://integrate.api.nvidia.com/v1)
  NIM_CLAUDE_HOME    Config directory (default ~/.nim-claude)
  CLAUDE_BIN         Override path to the claude binary

${c.bold("EXAMPLES")}
  ${c.dim("# one-time setup")}
  nim keys add nvapi-XXXXXXXXXXXXXXXX personal
  nim keys add nvapi-YYYYYYYYYYYYYYYY work

  ${c.dim("# launch claude — works exactly like running 'claude' directly")}
  nim
  nim --resume
  nim "explain this codebase"
`);
}

async function main() {
  const argv = process.argv.slice(2);
  const port = DEFAULT_PORT;
  const cmd = argv[0];
  try {
    if (cmd === "help" || cmd === "--help" || cmd === "-h") return printHelp();
    if (cmd === "status") return await cmdStatus();
    if (cmd === "keys") return await cmdKeys(argv.slice(1));
    if (cmd === "models") return await cmdModels(argv.slice(1));
    if (cmd === "rotation") return await cmdRotation(argv.slice(1));
    if (cmd === "use") return await cmdUse(argv.slice(1));
    if (cmd === "telemetry" || cmd === "stats") return await cmdTelemetry();
    if (cmd === "proxy") return await cmdProxyOnly(port);
    if (cmd === "version" || cmd === "--version" || cmd === "-v") {
      console.log("nim 0.1.0");
      return;
    }
    // Default + anything else → forward to claude
    await runClaudeWrapped(argv, port);
  } catch (e) {
    err(e instanceof Error ? e.message : String(e));
    process.exit(1);
  }
}

main();
