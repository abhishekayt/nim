import { pgTable, text, timestamp, integer, boolean, jsonb, serial, uuid, real, index, varchar } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const conversationsTable = pgTable("conversations", {
  id: uuid("id").primaryKey().defaultRandom(),
  externalId: text("external_id").notNull().unique(),
  modelName: text("model_name").notNull(),
  category: text("category"),
  messageCount: integer("message_count").notNull().default(0),
  totalTokens: integer("total_tokens").notNull().default(0),
  summary: text("summary"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
}, (table) => [
  index("conversations_external_id_idx").on(table.externalId),
]);

export const conversationMessagesTable = pgTable("conversation_messages", {
  id: uuid("id").primaryKey().defaultRandom(),
  conversationId: uuid("conversation_id").notNull().references(() => conversationsTable.id, { onDelete: "cascade" }),
  role: text("role").notNull(),
  content: text("content").notNull(),
  tokens: integer("tokens").default(0),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
}, (table) => [
  index("messages_conversation_id_idx").on(table.conversationId),
]);

export const telemetryTable = pgTable("telemetry", {
  id: uuid("id").primaryKey().defaultRandom(),
  ts: timestamp("ts", { withTimezone: true }).defaultNow().notNull(),
  modelName: text("model_name").notNull(),
  modelId: text("model_id"),
  keyId: text("key_id"),
  providerId: text("provider_id"),
  categories: text("categories").array().notNull().default([]),
  signals: text("signals").array().notNull().default([]),
  latencyMs: integer("latency_ms").notNull(),
  status: text("status").notNull(),
  cached: boolean("cached").notNull().default(false),
  streaming: boolean("streaming").notNull().default(false),
  inputTokens: integer("input_tokens"),
  outputTokens: integer("output_tokens"),
  errorMessage: text("error_message"),
  toolRetries: integer("tool_retries").default(0),
}, (table) => [
  index("telemetry_ts_idx").on(table.ts),
  index("telemetry_model_idx").on(table.modelName),
  index("telemetry_status_idx").on(table.status),
]);

export const modelRankingsTable = pgTable("model_rankings", {
  id: uuid("id").primaryKey().defaultRandom(),
  modelName: text("model_name").notNull().unique(),
  category: text("category").notNull(),
  successRate: real("success_rate").notNull().default(0),
  avgLatencyMs: integer("avg_latency_ms").notNull().default(0),
  avgToolRetryRate: real("avg_tool_retry_rate").notNull().default(0),
  totalRequests: integer("total_requests").notNull().default(0),
  score: real("score").notNull().default(0),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
}, (table) => [
  index("rankings_category_idx").on(table.category),
  index("rankings_score_idx").on(table.score),
]);

export const fileContextTable = pgTable("file_context", {
  id: uuid("id").primaryKey().defaultRandom(),
  projectPath: text("project_path").notNull(),
  filePath: text("file_path").notNull(),
  content: text("content").notNull(),
  fileType: text("file_type"),
  lastModified: timestamp("last_modified", { withTimezone: true }).defaultNow().notNull(),
  tokenCount: integer("token_count").default(0),
}, (table) => [
  index("file_ctx_project_path_idx").on(table.projectPath),
  index("file_ctx_file_path_idx").on(table.filePath),
]);

export const insertConversationSchema = createInsertSchema(conversationsTable).omit({ id: true, createdAt: true, updatedAt: true });
export const insertMessageSchema = createInsertSchema(conversationMessagesTable).omit({ id: true, createdAt: true });
export const insertTelemetrySchema = createInsertSchema(telemetryTable).omit({ id: true, ts: true });
export const insertModelRankingSchema = createInsertSchema(modelRankingsTable).omit({ id: true, updatedAt: true });
export const insertFileContextSchema = createInsertSchema(fileContextTable).omit({ id: true, lastModified: true });

export type InsertConversation = z.infer<typeof insertConversationSchema>;
export type InsertMessage = z.infer<typeof insertMessageSchema>;
export type InsertTelemetry = z.infer<typeof insertTelemetrySchema>;
export type InsertModelRanking = z.infer<typeof insertModelRankingSchema>;
export type InsertFileContext = z.infer<typeof insertFileContextSchema>;
export type Conversation = typeof conversationsTable.$inferSelect;
export type ConversationMessage = typeof conversationMessagesTable.$inferSelect;
export type Telemetry = typeof telemetryTable.$inferSelect;
export type ModelRanking = typeof modelRankingsTable.$inferSelect;
export type FileContext = typeof fileContextTable.$inferSelect;
