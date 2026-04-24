/**
 * Multi-sample + vote.
 *
 * For requests we believe are hard (classifier confidence low, category
 * `reasoning-hard`, or explicit caller request), we issue N parallel
 * non-stream calls with varied temperature/top_p and pick the best
 * response according to a judge strategy:
 *
 *   - "length"       — pick the longest non-empty answer (trivial fallback)
 *   - "verifier"     — for tool-bearing answers, pick the one whose
 *                      proposed file changes survive the verifier
 *   - "consensus"    — group by answer fingerprint (normalized), pick
 *                      the largest group; ties broken by length
 *
 * This is opt-in and rate-limited (max N=5) to keep cost bounded.
 */

import type { OpenAIRequest, OpenAINonStreamResponse } from "./translator";
import type { CallContext, CallOptions } from "./nim-client";
import { createHash } from "node:crypto";

export type JudgeStrategy = "length" | "consensus";

export interface MultiSampleResult {
  winner: { ctx: CallContext; data: OpenAINonStreamResponse; sampleIndex: number };
  samples: Array<{
    ctx: CallContext | null;
    data: OpenAINonStreamResponse | null;
    error: string | null;
    temperature: number;
    top_p: number;
  }>;
  strategy: JudgeStrategy;
  totalDurationMs: number;
}

interface SamplePlan {
  temperature: number;
  top_p: number;
}

function buildSamplePlan(n: number, base: { temperature?: number; top_p?: number }): SamplePlan[] {
  const baseT = base.temperature ?? 0.7;
  const baseP = base.top_p ?? 0.95;
  // First sample sticks to user's params; subsequent ones spread out.
  const plans: SamplePlan[] = [{ temperature: baseT, top_p: baseP }];
  const variants = [
    { dT: -0.3, dP: -0.05 },
    { dT: 0.2, dP: 0 },
    { dT: 0.4, dP: -0.1 },
    { dT: 0.5, dP: 0.04 },
  ];
  for (let i = 0; i < n - 1; i++) {
    const v = variants[i] ?? variants[variants.length - 1]!;
    plans.push({
      temperature: Math.min(1.5, Math.max(0, baseT + v.dT)),
      top_p: Math.min(1, Math.max(0.05, baseP + v.dP)),
    });
  }
  return plans;
}

function normalizeAnswer(data: OpenAINonStreamResponse): string {
  const choice = data.choices?.[0];
  if (!choice) return "";
  // For tool calls, fingerprint is the (sorted) list of (name, normalized args).
  const tcs = choice.message?.tool_calls;
  if (tcs && tcs.length > 0) {
    const parts = tcs.map((tc) => {
      let argsObj: unknown = null;
      try {
        argsObj = JSON.parse(tc.function.arguments || "{}");
      } catch {
        argsObj = tc.function.arguments;
      }
      return `${tc.function.name}::${JSON.stringify(argsObj)}`;
    });
    parts.sort();
    return parts.join("|");
  }
  // For text answers, normalize whitespace + lowercase first 1k chars.
  const txt = (choice.message?.content ?? "").trim().toLowerCase();
  return txt.replace(/\s+/g, " ").slice(0, 1000);
}

function answerLength(data: OpenAINonStreamResponse): number {
  const choice = data.choices?.[0];
  if (!choice) return 0;
  const tcs = choice.message?.tool_calls;
  if (tcs && tcs.length > 0) {
    return tcs.reduce(
      (acc, tc) => acc + tc.function.name.length + (tc.function.arguments?.length ?? 0),
      0,
    );
  }
  return (choice.message?.content ?? "").length;
}

function pickByLength(
  samples: Array<{ ctx: CallContext | null; data: OpenAINonStreamResponse | null }>,
): number {
  let best = -1;
  let bestLen = -1;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    if (!s.data) continue;
    const len = answerLength(s.data);
    if (len > bestLen) {
      best = i;
      bestLen = len;
    }
  }
  return best;
}

function pickByConsensus(
  samples: Array<{ ctx: CallContext | null; data: OpenAINonStreamResponse | null }>,
): number {
  const groups = new Map<string, number[]>();
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    if (!s.data) continue;
    const sig = createHash("sha256").update(normalizeAnswer(s.data)).digest("hex");
    const arr = groups.get(sig) ?? [];
    arr.push(i);
    groups.set(sig, arr);
  }
  if (groups.size === 0) return -1;
  let bestIdxs: number[] = [];
  let bestSize = 0;
  for (const idxs of groups.values()) {
    if (idxs.length > bestSize) {
      bestSize = idxs.length;
      bestIdxs = idxs;
    }
  }
  if (bestIdxs.length === 0) return -1;
  // Tie-break by longest answer in the winning group.
  let winner = bestIdxs[0]!;
  let winnerLen = -1;
  for (const i of bestIdxs) {
    const s = samples[i]!;
    if (!s.data) continue;
    const len = answerLength(s.data);
    if (len > winnerLen) {
      winnerLen = len;
      winner = i;
    }
  }
  return winner;
}

/**
 * Run N parallel samples and pick the best. Caller supplies the actual
 * model-call function so this module stays decoupled from nim-client's
 * import graph (avoids circular deps).
 */
export async function runMultiSample(
  basePayload: OpenAIRequest,
  requestedAnthropicModel: string,
  caller: (
    payload: OpenAIRequest,
    requestedModel: string,
    opts: CallOptions,
  ) => Promise<{ ctx: CallContext; data: OpenAINonStreamResponse }>,
  options: {
    n: number;
    strategy?: JudgeStrategy;
    callOpts?: CallOptions;
  },
): Promise<MultiSampleResult> {
  const start = Date.now();
  const n = Math.min(5, Math.max(2, options.n));
  const strategy = options.strategy ?? "consensus";
  const plans = buildSamplePlan(n, basePayload);

  const settled = await Promise.allSettled(
    plans.map((plan) =>
      caller(
        { ...basePayload, temperature: plan.temperature, top_p: plan.top_p, stream: false },
        requestedAnthropicModel,
        // Disable cache so we don't get N copies of the same cached answer.
        { ...(options.callOpts ?? {}), noCache: true },
      ),
    ),
  );

  const samples: MultiSampleResult["samples"] = settled.map((r, i) => {
    const plan = plans[i]!;
    if (r.status === "fulfilled") {
      return {
        ctx: r.value.ctx,
        data: r.value.data,
        error: null,
        temperature: plan.temperature,
        top_p: plan.top_p,
      };
    }
    return {
      ctx: null,
      data: null,
      error: r.reason instanceof Error ? r.reason.message : String(r.reason),
      temperature: plan.temperature,
      top_p: plan.top_p,
    };
  });

  let winnerIdx =
    strategy === "consensus" ? pickByConsensus(samples) : pickByLength(samples);
  if (winnerIdx < 0) winnerIdx = pickByLength(samples);
  if (winnerIdx < 0) {
    throw new Error(
      `multi-sample: all ${n} samples failed. First error: ${samples.find((s) => s.error)?.error ?? "unknown"}`,
    );
  }

  const w = samples[winnerIdx]!;
  if (!w.ctx || !w.data) {
    throw new Error("multi-sample: winning sample has no data (should not happen)");
  }

  return {
    winner: { ctx: w.ctx, data: w.data, sampleIndex: winnerIdx },
    samples,
    strategy,
    totalDurationMs: Date.now() - start,
  };
}
