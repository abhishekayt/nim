/**
 * Conversation caching with Supabase persistence.
 *
 * Implements sliding-window conversation caching keyed by conversation ID
 * to reduce latency on multi-turn conversations. This approximates Claude's
 * prompt caching behavior for NIM models that don't support it natively.
 */

import { db } from "@workspace/db";
import {
  conversationsTable,
  conversationMessagesTable,
  type InsertConversation,
  type InsertMessage,
} from "@workspace/db/schema";
import { eq, desc, and, sql } from "drizzle-orm";

interface CachedMessage {
  role: string;
  content: string;
  tokens: number;
}

interface ConversationCache {
  messages: CachedMessage[];
  modelName: string;
  category: string | null;
  totalTokens: number;
  summary: string | null;
}

const MAX_CONVERSATION_MESSAGES = 200;
const COMPACTION_THRESHOLD = 40; // Summarize after this many messages

function generateConversationId(req: { model: string; messages: Array<{ role: string; content: unknown }> }): string {
  // Use first user message + model as stable conversation identifier
  const firstUser = req.messages.find((m) => m.role === "user");
  const firstUserText = typeof firstUser?.content === "string"
    ? firstUser.content
    : JSON.stringify(firstUser?.content ?? "");
  const hash = Buffer.from(`${req.model}:${firstUserText.slice(0, 200)}`).toString("base64").replace(/[^a-zA-Z0-9]/g, "").slice(0, 32);
  return hash;
}

export async function getConversationCache(
  req: { model: string; messages: Array<{ role: string; content: unknown }> },
): Promise<ConversationCache | null> {
  const externalId = generateConversationId(req);

  const conv = await db
    .select()
    .from(conversationsTable)
    .where(eq(conversationsTable.externalId, externalId))
    .limit(1)
    .then((rows) => rows[0] ?? null);

  if (!conv) return null;

  const msgs = await db
    .select()
    .from(conversationMessagesTable)
    .where(eq(conversationMessagesTable.conversationId, conv.id))
    .orderBy(desc(conversationMessagesTable.createdAt))
    .limit(MAX_CONVERSATION_MESSAGES)
    .then((rows) =>
      rows.reverse().map((m) => ({
        role: m.role,
        content: m.content,
        tokens: m.tokens ?? 0,
      })),
    );

  return {
    messages: msgs,
    modelName: conv.modelName,
    category: conv.category,
    totalTokens: conv.totalTokens,
    summary: conv.summary,
  };
}

export async function appendToConversationCache(
  req: { model: string; messages: Array<{ role: string; content: unknown }> },
  responseContent: string,
  modelName: string,
  category: string | null,
  tokens: number,
): Promise<void> {
  const externalId = generateConversationId(req);

  let conv = await db
    .select()
    .from(conversationsTable)
    .where(eq(conversationsTable.externalId, externalId))
    .limit(1)
    .then((rows) => rows[0] ?? null);

  if (!conv) {
    const inserted = await db
      .insert(conversationsTable)
      .values({
        externalId,
        modelName,
        category,
        messageCount: 0,
        totalTokens: 0,
      } as InsertConversation)
      .returning();
    conv = inserted[0]!;
  }

  // Insert the last user message and assistant response
  const lastUserMsg = [...req.messages].reverse().find((m) => m.role === "user");
  const lastAssistantMsg = { role: "assistant", content: responseContent };

  const toInsert: InsertMessage[] = [];

  if (lastUserMsg) {
    const userText = typeof lastUserMsg.content === "string"
      ? lastUserMsg.content
      : JSON.stringify(lastUserMsg.content);
    toInsert.push({
      conversationId: conv.id,
      role: "user",
      content: userText,
      tokens: Math.ceil(userText.length / 4), // Rough token estimate
    });
  }

  toInsert.push({
    conversationId: conv.id,
    role: "assistant",
    content: responseContent,
    tokens,
  });

  await db.insert(conversationMessagesTable).values(toInsert);

  // Update conversation metadata
  const newTotalTokens = conv.totalTokens + tokens + toInsert.reduce((sum, m) => sum + (m.tokens ?? 0), 0);
  await db
    .update(conversationsTable)
    .set({
      messageCount: sql`${conversationsTable.messageCount} + ${toInsert.length}`,
      totalTokens: newTotalTokens,
      updatedAt: new Date(),
    })
    .where(eq(conversationsTable.id, conv.id));

  // Trigger compaction if needed
  const msgCount = await db
    .select({ count: sql<number>`count(*)` })
    .from(conversationMessagesTable)
    .where(eq(conversationMessagesTable.conversationId, conv.id))
    .then((rows) => rows[0]?.count ?? 0);

  if (msgCount > COMPACTION_THRESHOLD) {
    await compactConversation(conv.id);
  }
}

async function compactConversation(conversationId: string): Promise<void> {
  const msgs = await db
    .select()
    .from(conversationMessagesTable)
    .where(eq(conversationMessagesTable.conversationId, conversationId))
    .orderBy(conversationMessagesTable.createdAt);

  if (msgs.length <= COMPACTION_THRESHOLD) return;

  // Summarize the first half of messages
  const toSummarize = msgs.slice(0, Math.floor(msgs.length / 2));
  const summary = `Previous context: ${toSummarize.length} messages covering ${toSummarize.map((m) => `${m.role}: ${m.content.slice(0, 100)}...`).join("; ")}`;

  // Delete old messages and insert summary
  const idsToDelete = toSummarize.map((m) => m.id);
  await db.delete(conversationMessagesTable).where(
    and(
      eq(conversationMessagesTable.conversationId, conversationId),
      sql`${conversationMessagesTable.id} = ANY(${idsToDelete})`,
    ),
  );

  await db.insert(conversationMessagesTable).values({
    conversationId,
    role: "system",
    content: summary,
    tokens: Math.ceil(summary.length / 4),
  });

  await db
    .update(conversationsTable)
    .set({ summary, updatedAt: new Date() })
    .where(eq(conversationsTable.id, conversationId));
}

export async function clearConversationCache(externalId?: string): Promise<void> {
  if (externalId) {
    const conv = await db
      .select()
      .from(conversationsTable)
      .where(eq(conversationsTable.externalId, externalId))
      .limit(1)
      .then((rows) => rows[0] ?? null);
    if (conv) {
      await db.delete(conversationsTable).where(eq(conversationsTable.id, conv.id));
    }
  } else {
    await db.delete(conversationMessagesTable).where(sql`true`);
    await db.delete(conversationsTable).where(sql`true`);
  }
}

export async function getCacheStats(): Promise<{ conversations: number; totalMessages: number; totalTokens: number }> {
  const convs = await db.select({ count: sql<number>`count(*)` }).from(conversationsTable).then((r) => r[0]?.count ?? 0);
  const msgs = await db.select({ count: sql<number>`count(*)` }).from(conversationMessagesTable).then((r) => r[0]?.count ?? 0);
  const tokens = await db.select({ sum: sql<number>`coalesce(sum(total_tokens), 0)` }).from(conversationsTable).then((r) => r[0]?.sum ?? 0);
  return { conversations: convs, totalMessages: msgs, totalTokens: tokens };
}
