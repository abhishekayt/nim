const MAX_RECORDS = Number(process.env["NIM_TELEMETRY_MAX"] ?? 200);

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
}

const records: TelemetryRecord[] = [];

export function recordRequest(rec: TelemetryRecord): void {
  records.push(rec);
  if (records.length > MAX_RECORDS) records.splice(0, records.length - MAX_RECORDS);
}

export function getTelemetry(limit = 50): { records: TelemetryRecord[]; summary: TelemetrySummary } {
  const slice = records.slice(-limit).reverse();
  return { records: slice, summary: summarize() };
}

export interface ModelStat {
  modelName: string;
  count: number;
  errors: number;
  cached: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
}

export interface TelemetrySummary {
  total: number;
  errors: number;
  cached: number;
  cacheHitRate: number;
  avgLatencyMs: number;
  perModel: ModelStat[];
}

function summarize(): TelemetrySummary {
  const total = records.length;
  if (total === 0) {
    return { total: 0, errors: 0, cached: 0, cacheHitRate: 0, avgLatencyMs: 0, perModel: [] };
  }
  let errors = 0; let cached = 0; let totalLatency = 0;
  const byModel = new Map<string, { count: number; errors: number; cached: number; latencies: number[]; tokIn: number; tokOut: number }>();
  for (const r of records) {
    if (r.status === "error") errors++;
    if (r.cached) cached++;
    totalLatency += r.latencyMs;
    let m = byModel.get(r.modelName);
    if (!m) { m = { count: 0, errors: 0, cached: 0, latencies: [], tokIn: 0, tokOut: 0 }; byModel.set(r.modelName, m); }
    m.count++;
    if (r.status === "error") m.errors++;
    if (r.cached) m.cached++;
    m.latencies.push(r.latencyMs);
    m.tokIn += r.inputTokens ?? 0;
    m.tokOut += r.outputTokens ?? 0;
  }
  const perModel: ModelStat[] = Array.from(byModel.entries()).map(([modelName, s]) => {
    const sorted = s.latencies.slice().sort((a, b) => a - b);
    const avg = sorted.reduce((acc, v) => acc + v, 0) / sorted.length;
    const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ?? 0;
    return {
      modelName, count: s.count, errors: s.errors, cached: s.cached,
      avgLatencyMs: Math.round(avg), p95LatencyMs: Math.round(p95),
      totalInputTokens: s.tokIn, totalOutputTokens: s.tokOut,
    };
  }).sort((a, b) => b.count - a.count);
  return {
    total, errors, cached,
    cacheHitRate: total > 0 ? cached / total : 0,
    avgLatencyMs: Math.round(totalLatency / total),
    perModel,
  };
}

export function clearTelemetry(): void { records.length = 0; }
