-- Supabase Schema Setup for Passos Magicos ML Pipeline
-- Run this SQL in the Supabase SQL Editor to create the required tables.

-- Table: predictions_log
-- Stores every prediction made by the API
CREATE TABLE IF NOT EXISTS predictions_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    input_data JSONB,
    prediction INTEGER,
    risk_level VARCHAR(10),
    probability_at_risk FLOAT,
    model_type VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying by timestamp
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions_log (timestamp DESC);

-- Table: model_metrics
-- Stores evaluation metrics for each model version
CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    model_type VARCHAR(100),
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table: drift_reports
-- Stores drift detection results
CREATE TABLE IF NOT EXISTS drift_reports (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    drift_detected BOOLEAN,
    drifted_features JSONB,
    n_features_analyzed INTEGER,
    n_features_drifted INTEGER,
    report_path VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE predictions_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE drift_reports ENABLE ROW LEVEL SECURITY;

-- Create policies for service role access
CREATE POLICY "Allow service role full access on predictions_log"
    ON predictions_log FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role full access on model_metrics"
    ON model_metrics FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role full access on drift_reports"
    ON drift_reports FOR ALL
    USING (true)
    WITH CHECK (true);
