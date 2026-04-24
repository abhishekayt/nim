/**
 * Telemetry-driven model ranking with Supabase persistence.
 *
 * Replaces the in-memory telemetry store with a database-backed system
 * that learns from historical performance to dynamically rank models.
 */

import { db } from "@workspace/db";
import {
  telemetryTable,
  modelRankingsTable,
  type InsertTelemetry,
  type InsertModelRanking,
} from "@workspace/db/schema";
import { eq, desc, sql, and, gte } from "drizzle-orm";

export interface TelemetryRecord {
  ts: number;
  modelName: string;
  modelId: string | null;
  keyId: string | null;
  providerId: string | null;
  categories: string[];
  signals: string[];
  latencyMs: number;
  status: "ok" | "error" | "cached";
  cached: boolean;
  streaming: boolean;
  inputTokens: number | null;
  outputTokens: number | null;
  errorMessage: string | null;
  toolRetries?: number;
}

export interface ModelStat {
  modelName: string;
  category: string;
  count: number;
  errors: number;
  cached: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  successRate: number;
  toolRetryRate: number;
  score: number;
}

export async function recordRequestDb(rec: TelemetryRecord): Promise<void> {
  await db.insert(telemetryTable).values({
    modelName: rec.modelName,
    modelId: rec.modelId,
    keyId: rec.keyId,
    providerId: rec.providerId,
    categories: rec.categories,
    signals: rec.signals,
    latencyMs: rec.latencyMs,
    status: rec.status,
    cached: rec.cached,
    streaming: rec.streaming,
    inputTokens: rec.inputTokens,
    outputTokens: rec.outputTokens,
    errorMessage: rec.errorMessage,
    toolRetries: rec.toolRetries ?? 0,
  } as InsertTelemetry);
}

export async function getTelemetryDb(limit = 100): Promise<{ records: TelemetryRecord[]; summary: TelemetrySummary }> {
  const rows = await db
    .select()
    .from(telemetryTable)
    .orderBy(desc(telemetryTable.ts))
    .limit(limit);

  const records: TelemetryRecord[] = rows.map((r) => ({
    ts: new Date(r.ts).getTime(),
    modelName: r.modelName,
    modelId: r.modelId,
    keyId: r.keyId,
    providerId: r.providerId,
    categories: r.categories,
    signals: r.signals,
    latencyMs: r.latencyMs,
    status: r.status as "ok" | "error" | "cached",
    cached: r.cached,
    streaming: r.streaming,
    inputTokens: r.inputTokens,
    outputTokens: r.outputTokens,
    errorMessage: r.errorMessage,
    toolRetries: r.toolRetries ?? 0,
  }));

  return { records, summary: await summarize() };
}

export interface TelemetrySummary {
  total: number;
  errors: number;
  cached: number;
  cacheHitRate: number;
  avgLatencyMs: number;
  perModel: ModelStat[];
}

async function summarize(): Promise<TelemetrySummary> {
  const total = await db.select({ count: sql<number>`count(*)` }).from(telemetryTable).then((r) => r[0]?.count ?? 0);

  if (total === 0) {
    return { total: 0, errors: 0, cached: 0, cacheHitRate: 0, avgLatencyMs: 0, perModel: [] };
  }

  const stats = await db
    .select({
      modelName: telemetryTable.modelName,
      count: sql<number>`count(*)`,
      errors: sql<number>`sum(case when ${telemetryTable.status} = 'error' then 1 else 0 end)`,
      cached: sql<number>`sum(case when ${telemetryTable.cached} = true then 1 else 0 end)`,
      avgLatency: sql<number>`round(avg(${telemetryTable.latencyMs}))`,
      p95Latency: sql<number>`percentile_cont(0.95) within group (order by ${telemetryTable.latencyMs})`,
      totalInput: sql<number>`coalesce(sum(${telemetryTable.inputTokens}), 0)`,
      totalOutput: sql<number>`coalesce(sum(${telemetryTable.outputTokens}), 0)`,
      toolRetries: sql<number>`coalesce(sum(${telemetryTable.toolRetries}), 0)`,
    })
    .from(telemetryTable)
    .groupBy(telemetryTable.modelName);

  const perModel: ModelStat[] = stats.map((s) => {
    const successRate = s.count > 0 ? (s.count - s.errors) / s.count : 0;
    const toolRetryRate = s.count > 0 ? (s.toolRetries ?? 0) / s.count : 0;
    return {
      modelName: s.modelName,
      category: "general",
      count: s.count,
      errors: s.errors,
      cached: s.cached,
      avgLatencyMs: s.avgLatency ?? 0,
      p95LatencyMs: Math.round(s.p95Latency ?? 0),
      totalInputTokens: s.totalInput,
      totalOutputTokens: s.totalOutput,
      successRate,
      toolRetryRate,
      score: calculateScore(successRate, s.avgLatency ?? 0, toolRetryRate),
    };
  });

  const errors = perModel.reduce((sum, m) => sum + m.errors, 0);
  const cached = perModel.reduce((sum, m) => sum + m.cached, 0);
  const avgLatency = perModel.reduce((sum, m) => sum + m.avgLatencyMs * m.count, 0) / total;

  return {
    total,
    errors,
    cached,
    cacheHitRate: total > 0 ? cached / total : 0,
    avgLatencyMs: Math.round(avgLatency),
    perModel: perModel.sort((a, b) => b.score - a.score),
  };
}

function calculateScore(successRate: number, avgLatencyMs: number, toolRetryRate: number): number {
  // Normalize latency: lower is better, cap at 30s
  const latencyScore = Math.max(0, 1 - avgLatencyMs / 30000);
  // Success rate: higher is better
  const successScore = successRate;
  // Tool retry rate: lower is better
  const toolScore = Math.max(0, 1 - toolRetryRate * 5);

  // Weighted composite
  return successScore * 0.5 + latencyScore * 0.3 + toolScore * 0.2;
}

export async function updateModelRankings(): Promise<void> {
  const summary = await summarize();

  for (const stat of summary.perModel) {
    const existing = await db
      .select()
      .from(modelRankingsTable)
      .where(eq(modelRankingsTable.modelName, stat.modelName))
      .limit(1)
      .then((r) => r[0] ?? null);

    if (existing) {
      await db
        .update(modelRankingsTable)
        .set({
          successRate: stat.successRate,
          avgLatencyMs: stat.avgLatencyMs,
          avgToolRetryRate: stat.toolRetryRate,
          totalRequests: stat.count,
          score: stat.score,
          updatedAt: new Date(),
        })
        .where(eq(modelRankingsTable.id, existing.id));
    } else {
      await db.insert(modelRankingsTable).values({
        modelName: stat.modelName,
        category: stat.category,
        successRate: stat.successRate,
        avgLatencyMs: stat.avgLatencyMs,
        avgToolRetryRate: stat.toolRetryRate,
        totalRequests: stat.count,
        score: stat.score,
      } as InsertModelRanking);
    }
  }
}

export async function getModelRankings(category?: string): Promise<ModelStat[]> {
  const query = category
    ? db.select().from(modelRankingsTable).where(eq(modelRankingsTable.category, category))
    : db.select().from(modelRankingsTable);

  const rows = await query.orderBy(desc(modelRankingsTable.score));

  return rows.map((r) => ({
    modelName: r.modelName,
    category: r.category,
    count: r.totalRequests,
    errors: 0,
    cached: 0,
    avgLatencyMs: r.avgLatencyMs,
    p95LatencyMs: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
    successRate: r.successRate,
    toolRetryRate: r.avgToolRetryRate,
    score: r.score,
  }));
}

export async function clearTelemetryDb(): Promise<void> {
  await db.delete(telemetryTable).where(sql`true`);
}

// Re-export for compatibility
export { summarize };
