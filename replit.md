# Claude Code → NVIDIA NIM Wrapper

## Overview

A drop-in Anthropic-compatible proxy that lets Claude Code (and any other Anthropic-API client) talk to NVIDIA NIM's free OpenAI-compatible models. Manages up to 3 NIM API keys with automatic rotation on rate limits, and a list of fallback models with automatic switching on errors.

## How users connect Claude Code

1. Open the dashboard at the deployed URL.
2. Add 1–3 NVIDIA NIM API keys (`nvapi-…`).
3. Set in their shell:
   - `ANTHROPIC_BASE_URL=<deployed url>`
   - `ANTHROPIC_API_KEY=anything-non-empty` (Claude Code requires it; the proxy ignores it)
4. Run `claude`.

## Packages

- **`nim-cli/`** — Standalone, globally installable CLI. After `cd nim-cli && npm install -g .`, the `nim` command works anywhere on the user's machine and behaves exactly like `claude`, but routes traffic through the embedded NIM proxy. Config lives at `~/.nim-claude/config.json`. **This is what end-users install.**
- **`artifacts/api-server/`** — The same proxy as a hosted web service (with a dashboard at `/`), useful when running on Replit or as a remote shared proxy that multiple machines point `ANTHROPIC_BASE_URL` at.

Both share the same proxy logic (translator, streaming, router, key/model rotation, dashboard HTML).

## Architecture

- `artifacts/api-server` — Express server that hosts everything:
  - `POST /v1/messages` — Anthropic Messages API endpoint, translates to NIM's OpenAI chat completions, supports streaming and tool use.
  - `GET /v1/models` — list of configured models.
  - `GET /admin/status`, `/admin/keys`, `/admin/models`, `/admin/rotation` — config CRUD for the dashboard.
  - `GET /` — the dashboard (vanilla HTML/JS).
- `src/proxy/store.ts` — file-backed config (`.data/config.json`) with env-var bootstrap (`NIM_API_KEY_1..3`, `NIM_BASE_URL`).
- `src/proxy/translator.ts` — Anthropic ↔ OpenAI request/response translation (text, images, tool_use, tool_result).
- `src/proxy/streaming.ts` — translates OpenAI SSE chunks into Anthropic `message_start` / `content_block_*` / `message_delta` / `message_stop` events.
- `src/proxy/router.ts` — key/model selection with cooldowns (5 min for keys on 429, 1 min for models on errors).
- `src/proxy/nim-client.ts` — calls NIM, retries across keys/models on rate-limit / model errors.

### Tier 1 + Tier 2 accuracy modules (added in `src/proxy/`)

- `promptCache.ts` — parses Anthropic `cache_control: { type: "ephemeral" }` markers in `system`/`messages`/`tools`; produces stable prefix hashes for cache keys.
- `toolValidator.ts` — Ajv-backed JSON Schema validation of tool-call arguments with auto-repair (JSON heal → type coerce → fill empty required strings); per-model tool-format presets.
- `verifier.ts` — opt-in post-edit verifier that runs configured `tsc --noEmit` / `eslint` / test commands (env vars `NIM_VERIFY_TYPECHECK_CMD`, `NIM_VERIFY_LINT_CMD`, `NIM_VERIFY_TEST_CMD`, `NIM_VERIFY_ON_EDIT=1`); time-budgeted (~20s).
- `cascade.ts` — given classifier categories + confidence, builds an ordered escalation plan over the `coding` / `reasoning` / `vision` / `general` lanes; cheap-first when confidence is low, strong-first when `preferAccuracy` is set.
- `summarizer.ts` (rewritten) — deterministic: keeps `tool_use`/`tool_result` pairs atomic, preserves fenced code blocks, dedupes repeated `Read` of the same file, keeps last N turns verbatim, extractive summary of the dropped middle.
- `streamingCache.ts` — buffers SSE chunks during a stream, replays them on a later cache hit via `setImmediate` with original inter-chunk delays.
- `embeddings.ts` + `semanticCache.ts` — embeddings via NIM `nv-embedqa`; on exact-cache miss, top-1 cosine over recent N entries with the same tools-fingerprint, threshold default 0.92; telemetry status `cached-semantic`.
- `requestAugmenter.ts` (rewritten) — file-ref injection plus regex-based symbol extraction, shallow TS/JS import-graph walk, project tree at depth 3 (with file-count caps), and embedding-based file ranking when embeddings are configured.
- `multiSample.ts` — `runMultiSample(payload, n, judgeStrategy)` issues N parallel non-stream calls with varied temperature/top_p (`length` / `verifier` / `judge-llm`); used only for `reasoning` under low confidence (cap N=5, never on streams).

Wired into `src/routes/messages.ts` (non-stream: prompt-cache key, semantic-cache, cascade plan, opt-in verifier on edit, opt-in multi-sample) and `src/proxy/nim-client.ts` (semantic + exact cache, 3-pass `repairResponseToolCalls`, cascade tier escalation, streaming-cache wrap/replay).

New admin endpoints in `src/routes/admin.ts`: `/admin/semantic-cache/clear`, `/admin/streaming-cache/clear`, `/admin/verifier/status`, `/admin/verifier/run`, `/admin/cascade/config`. The `/admin/telemetry` response now also includes `semanticCache`, `streamingCache`, `cascade`, and `verifier` blocks.

New `.nimrc` (project-config) flags: `preferAccuracy: boolean`, `multiSample: boolean | number`.

## Stack

- Node 24, Express 5, pino logging
- esbuild bundle into `dist/index.mjs`
- Static dashboard copied from `src/public/` to `dist/public/` at build time
- `ajv` + `ajv-formats` for JSON Schema validation of tool-call args

## Key Commands

- `pnpm --filter @workspace/api-server run dev` — dev (build + start)
- `pnpm --filter @workspace/api-server run build` — production build
- `pnpm --filter @workspace/api-server run typecheck` — TypeScript check (see "Known pre-existing typecheck issues" below)

## Known pre-existing typecheck issues (NOT introduced by Tier 1/2 work)

These errors pre-date the accuracy work and are out of scope:

- `lib/db` and `lib/api-zod` are configured with `composite: true; emitDeclarationOnly: true` but have no build script, so their `dist/*.d.ts` files don't exist. This causes `TS6305` in `src/proxy/conversationCache.ts`, `src/proxy/telemetryDb.ts`, `src/routes/health.ts`. The downstream implicit-`any` errors in `conversationCache.ts` and `telemetryDb.ts` (drizzle row callbacks like `r => …`, `m => …`) are caused by the missing `.d.ts`, not by their own source.
- `lib/db/src/schema/index.ts` has `TS2344` errors against `drizzle-zod`'s `ZodType` constraint (zod-version mismatch, 5 errors).
- `src/proxy/store.ts` has a pre-existing nullable bug in the `KeyState[]` filter (`TS2322` + `TS2677`).

The Tier 1+2 modules themselves typecheck clean (no errors in any of: `promptCache.ts`, `toolValidator.ts`, `verifier.ts`, `cascade.ts`, `summarizer.ts`, `streamingCache.ts`, `embeddings.ts`, `semanticCache.ts`, `multiSample.ts`, `requestAugmenter.ts`, the new admin endpoints, the wiring in `messages.ts` and `nim-client.ts`).
