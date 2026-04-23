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

## Stack

- Node 24, Express 5, pino logging
- esbuild bundle into `dist/index.mjs`
- Static dashboard copied from `src/public/` to `dist/public/` at build time

## Key Commands

- `pnpm --filter @workspace/api-server run dev` — dev (build + start)
- `pnpm --filter @workspace/api-server run build` — production build
