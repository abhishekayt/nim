# nim — Claude Code wrapped with NVIDIA NIM

Drop-in middleware that lets you run **Claude Code** against **NVIDIA NIM's free models**, with automatic key rotation (up to 3 keys) and model fallback.

You get the entire Claude Code experience — UI, slash commands, tool use, file editing, everything — with `nim` as a transparent wrapper that injects the proxy.

## Install

Requires Node.js ≥ 18.17 and [Claude Code](https://docs.claude.com/en/docs/claude-code) installed.

From this repo:

```bash
cd nim-cli
npm install
npm run build
npm install -g .
```

Or with pnpm:

```bash
cd nim-cli
pnpm install
pnpm run build
pnpm link --global
```

That installs the `nim` command globally. Verify with:

```bash
nim --version
nim help
```

## Use

```bash
# 1. add up to 3 NVIDIA NIM API keys (get them at build.nvidia.com)
nim keys add nvapi-XXXXXXXXXXXXXXXX personal
nim keys add nvapi-YYYYYYYYYYYYYYYY work

# 2. just run `nim` instead of `claude` — same UI, same commands, same flags
nim
nim --resume
nim "explain this codebase"
nim mcp list
```

`nim` forwards every argument to `claude` after spinning up the local proxy on `127.0.0.1:8787`. From your perspective it *is* Claude Code.

## How rotation works

- **Auto key switching** *(default)* — On a `429` rate-limit response, the offending key is parked for 5 minutes and the next available key takes over for the rest of the session.
- **Auto model switching** *(default)* — On a `4xx` model error or `5xx` upstream error, the current model is parked for 1 minute and the next enabled model from the fallback list is used. Six free NIM models are pre-loaded.
- **Manual mode** — Pin to a specific key or model. Errors are recorded, but no rotation happens.

## CLI commands

```text
nim                                Launch claude through the proxy (default)
nim proxy                          Run only the proxy + dashboard at http://127.0.0.1:8787/
nim status                         Show keys, models, rotation, hit/error counts
nim keys list
nim keys add <nvapi-…> [label]
nim keys remove   <id>
nim keys enable   <id>
nim keys disable  <id>
nim keys clear    <id>             Clear a key's rate-limit cooldown
nim models list
nim models add    <model-name>
nim models remove <id>
nim models enable|disable <id>
nim models clear  <id>             Clear a model's failure cooldown
nim rotation key   auto|manual
nim rotation model auto|manual
nim use key   <id>                 Set the active key
nim use model <id>                 Set the active model
nim help
```

## Dashboard

A live web dashboard is available at `http://127.0.0.1:8787/` whenever the proxy is running (e.g. while `nim` is launched, or via `nim proxy`). Use it to add/remove keys and models, toggle rotation modes, and watch live status.

## Environment variables

| Variable             | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `NIM_PORT`           | Local proxy port (default `8787`)                                  |
| `NIM_API_KEY_1/2/3`  | Bootstrap up to 3 NIM keys on first run                            |
| `NIM_BASE_URL`       | NIM endpoint (default `https://integrate.api.nvidia.com/v1`)       |
| `NIM_CLAUDE_HOME`    | Config directory (default `~/.nim-claude`)                         |
| `CLAUDE_BIN`         | Override path to the `claude` binary                               |

Config (keys, models, rotation state) is persisted to `~/.nim-claude/config.json`.

## Pre-loaded fallback models

- `meta/llama-3.3-70b-instruct`
- `nvidia/llama-3.1-nemotron-70b-instruct`
- `meta/llama-3.1-405b-instruct`
- `qwen/qwen2.5-coder-32b-instruct`
- `mistralai/mixtral-8x22b-instruct-v0.1`
- `deepseek-ai/deepseek-r1`

Add or remove with `nim models add/remove`.

## Uninstall

```bash
npm uninstall -g nim-cli
# or, if you used pnpm link:
pnpm unlink --global nim-cli
```
