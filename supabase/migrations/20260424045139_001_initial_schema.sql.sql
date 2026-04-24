/*
  # Initial NIM Proxy Schema

  1. New Tables
    - `conversations` - Stores conversation metadata for caching and context management
    - `conversation_messages` - Stores individual messages per conversation
    - `telemetry` - Persistent telemetry data for model performance tracking
    - `model_rankings` - Dynamic model scoring per category for intelligent routing
    - `file_context` - Cached file contents for request augmentation

  2. Security
    - Enable RLS on all tables
    - Add service-role policies for server-side access
    - No direct client access needed; all access goes through the API server

  3. Indexes
    - External ID lookup for conversations
    - Conversation ID lookup for messages
    - Time-series indexes on telemetry
    - Category and score indexes on rankings
    - Project path indexes on file context
*/

CREATE TABLE IF NOT EXISTS conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id text NOT NULL UNIQUE,
  model_name text NOT NULL,
  category text,
  message_count integer NOT NULL DEFAULT 0,
  total_tokens integer NOT NULL DEFAULT 0,
  summary text,
  created_at timestamptz DEFAULT now() NOT NULL,
  updated_at timestamptz DEFAULT now() NOT NULL
);

CREATE INDEX IF NOT EXISTS conversations_external_id_idx ON conversations(external_id);

CREATE TABLE IF NOT EXISTS conversation_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role text NOT NULL,
  content text NOT NULL,
  tokens integer DEFAULT 0,
  created_at timestamptz DEFAULT now() NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_conversation_id_idx ON conversation_messages(conversation_id);

CREATE TABLE IF NOT EXISTS telemetry (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ts timestamptz DEFAULT now() NOT NULL,
  model_name text NOT NULL,
  model_id text,
  key_id text,
  provider_id text,
  categories text[] NOT NULL DEFAULT '{}',
  signals text[] NOT NULL DEFAULT '{}',
  latency_ms integer NOT NULL,
  status text NOT NULL,
  cached boolean NOT NULL DEFAULT false,
  streaming boolean NOT NULL DEFAULT false,
  input_tokens integer,
  output_tokens integer,
  error_message text,
  tool_retries integer DEFAULT 0
);

CREATE INDEX IF NOT EXISTS telemetry_ts_idx ON telemetry(ts);
CREATE INDEX IF NOT EXISTS telemetry_model_idx ON telemetry(model_name);
CREATE INDEX IF NOT EXISTS telemetry_status_idx ON telemetry(status);

CREATE TABLE IF NOT EXISTS model_rankings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name text NOT NULL UNIQUE,
  category text NOT NULL,
  success_rate real NOT NULL DEFAULT 0,
  avg_latency_ms integer NOT NULL DEFAULT 0,
  avg_tool_retry_rate real NOT NULL DEFAULT 0,
  total_requests integer NOT NULL DEFAULT 0,
  score real NOT NULL DEFAULT 0,
  updated_at timestamptz DEFAULT now() NOT NULL
);

CREATE INDEX IF NOT EXISTS rankings_category_idx ON model_rankings(category);
CREATE INDEX IF NOT EXISTS rankings_score_idx ON model_rankings(score);

CREATE TABLE IF NOT EXISTS file_context (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_path text NOT NULL,
  file_path text NOT NULL,
  content text NOT NULL,
  file_type text,
  last_modified timestamptz DEFAULT now() NOT NULL,
  token_count integer DEFAULT 0
);

CREATE INDEX IF NOT EXISTS file_ctx_project_path_idx ON file_context(project_path);
CREATE INDEX IF NOT EXISTS file_ctx_file_path_idx ON file_context(file_path);

-- Enable RLS
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE telemetry ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_rankings ENABLE ROW LEVEL SECURITY;
ALTER TABLE file_context ENABLE ROW LEVEL SECURITY;

-- Service role access policies (server-side only)
CREATE POLICY "Service role full access on conversations"
  ON conversations FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Service role full access on conversation_messages"
  ON conversation_messages FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Service role full access on telemetry"
  ON telemetry FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Service role full access on model_rankings"
  ON model_rankings FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Service role full access on file_context"
  ON file_context FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);
