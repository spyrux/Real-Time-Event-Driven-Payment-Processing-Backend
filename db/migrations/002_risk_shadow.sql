-- Apply once to an existing database; also safe to re-run.
BEGIN;
ALTER TABLE payments ADD COLUMN IF NOT EXISTS currency TEXT NOT NULL DEFAULT 'unknown';
CREATE INDEX IF NOT EXISTS payments_risk_history_idx
    ON payments (user_id, currency, created_at);
CREATE TABLE IF NOT EXISTS risk_assessments (
    event_id TEXT PRIMARY KEY,
    payment_id TEXT,
    source TEXT NOT NULL,
    assessment JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS risk_assessments_payment_idx
    ON risk_assessments (payment_id, created_at DESC);
COMMIT;
