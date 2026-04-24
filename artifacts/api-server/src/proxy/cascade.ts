/**
 * Cascade routing.
 *
 * Many requests are easy enough that a small/cheap model can handle them
 * correctly; only when the cheap model's answer fails downstream
 * validation (tool call malformed, verifier reports errors, classifier
 * confidence low) do we escalate to a larger / more capable model in the
 * same lane.
 *
 * The classifier in this proxy emits a small set of model lanes
 * (`coding`, `reasoning`, `vision`, `general`); the cheap-vs-strong split
 * happens via the *order* of categories the cascade emits. The
 * model-router (`router.ts`) walks each tier in order and within a tier
 * picks the next eligible model from `cfg.models` — so by listing the
 * primary lane first and then a fallback lane, we get a real escalation
 * across model strengths even though we only have 4 lanes.
 */

import type { ModelCategory } from "./store";

export interface CascadePlan {
  /** Ordered list of categories to try, cheapest first. */
  tiers: ModelCategory[];
  /** Reason the plan looks the way it does (for telemetry / dashboard). */
  reason: string;
  /** Whether the plan started with a downgrade (cheap-first attempt). */
  startedCheap: boolean;
}

interface PlanInput {
  /** Categories the classifier picked for this request. */
  categories: readonly ModelCategory[];
  /** Confidence (0..1) the classifier reported. */
  confidence: number;
  /** Whether the request includes tools (agentic loop). */
  hasTools: boolean;
  /** Whether the request includes images (forces vision tier). */
  hasImages: boolean;
  /** User override: prefer accuracy over cost (always start strong). */
  preferAccuracy?: boolean;
}

const CONFIDENCE_THRESHOLD = 0.55;

/**
 * Pick a sensible escalation lane for a primary category. We escalate
 * "coding" → "reasoning" (a reasoning model often handles edge-case
 * agentic logic better) and "general" → "reasoning" (free-form questions
 * benefit from chain-of-thought).
 */
const FALLBACK_LANE: Record<ModelCategory, ModelCategory[]> = {
  vision: [],            // vision must stay vision; no graceful fallback
  coding: ["reasoning"], // when coding fails, harder-thinking model
  reasoning: ["coding"], // when reasoning fails on a code task, swap
  general: ["reasoning"], // for harder general questions
};

/**
 * Build the escalation plan for a request.
 */
export function buildCascadePlan(input: PlanInput): CascadePlan {
  // Vision is non-negotiable: if there are images the model MUST be
  // vision-capable, so no cheap-first.
  if (input.hasImages) {
    return {
      tiers: ["vision"],
      reason: "vision-required",
      startedCheap: false,
    };
  }

  const primary: ModelCategory =
    (input.categories[0] as ModelCategory | undefined) ?? "general";

  // User opted for accuracy → put the strongest lane first.
  if (input.preferAccuracy) {
    const fb = FALLBACK_LANE[primary] ?? [];
    // For "coding" / "general", escalate to "reasoning" upfront.
    const strongFirst: ModelCategory[] =
      primary === "coding" || primary === "general"
        ? ["reasoning", primary]
        : [primary, ...fb];
    return {
      tiers: dedupe(strongFirst),
      reason: "user-prefers-accuracy",
      startedCheap: false,
    };
  }

  // High-confidence reasoning: skip cheap attempt.
  if (primary === "reasoning" && input.confidence >= 0.8) {
    return {
      tiers: ["reasoning"],
      reason: "reasoning-confident",
      startedCheap: false,
    };
  }

  // Confident enough to start with the primary tier; escalate only on
  // failure.
  if (input.confidence >= CONFIDENCE_THRESHOLD) {
    const fb = FALLBACK_LANE[primary] ?? [];
    return {
      tiers: dedupe([primary, ...fb]),
      reason: `confident-primary (${input.confidence.toFixed(2)})`,
      startedCheap: false,
    };
  }

  // Low confidence: lean on the cheaper general lane first when there
  // are no tools; for tool-bearing requests we still need a coding lane
  // up front (chat-only general models routinely botch tool calls).
  if (input.hasTools) {
    return {
      tiers: dedupe(["coding", "reasoning"]),
      reason: `low-confidence-tools (${input.confidence.toFixed(2)})`,
      startedCheap: primary !== "coding",
    };
  }

  return {
    tiers: dedupe(["general", "reasoning"]),
    reason: `low-confidence-general (${input.confidence.toFixed(2)})`,
    startedCheap: primary !== "general",
  };
}

function dedupe<T>(arr: T[]): T[] {
  const seen = new Set<T>();
  const out: T[] = [];
  for (const x of arr) {
    if (seen.has(x)) continue;
    seen.add(x);
    out.push(x);
  }
  return out;
}

/**
 * Telemetry: record a cascade event so the dashboard can show how often
 * we escalate and how often the cheap tier wins.
 */
interface CascadeEvent {
  at: number;
  plan: CascadePlan;
  /** 0-indexed tier that ultimately succeeded (-1 if all failed). */
  winningTier: number;
  /** Total number of upstream calls used (>= 1). */
  attempts: number;
}

const RECENT_MAX = 100;
const recent: CascadeEvent[] = [];

export function recordCascadeEvent(ev: Omit<CascadeEvent, "at">): void {
  recent.push({ ...ev, at: Date.now() });
  if (recent.length > RECENT_MAX) recent.shift();
}

export function getCascadeStats(): {
  total: number;
  cheapWins: number;
  escalations: number;
  avgAttempts: number;
  recent: CascadeEvent[];
} {
  if (recent.length === 0) {
    return { total: 0, cheapWins: 0, escalations: 0, avgAttempts: 0, recent: [] };
  }
  let cheapWins = 0;
  let escalations = 0;
  let totalAttempts = 0;
  for (const ev of recent) {
    if (ev.plan.startedCheap && ev.winningTier === 0) cheapWins++;
    if (ev.winningTier > 0) escalations++;
    totalAttempts += ev.attempts;
  }
  return {
    total: recent.length,
    cheapWins,
    escalations,
    avgAttempts: totalAttempts / recent.length,
    recent: recent.slice(-20),
  };
}
