-- Metadata schema for Spark quantum simulator
-- Uses DuckDB for metadata storage (WAL, checkpoints)

-- Write-ahead log for batch operations
CREATE TABLE IF NOT EXISTS wal (
    wal_id INTEGER PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    gate_start INTEGER NOT NULL,      -- Start of gate sequence (inclusive)
    gate_end INTEGER NOT NULL,        -- End of gate sequence (exclusive)
    state_version_in INTEGER NOT NULL,
    state_version_out INTEGER NOT NULL,
    status VARCHAR NOT NULL CHECK (status IN ('PENDING', 'COMMITTED', 'FAILED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    committed_at TIMESTAMP
);

-- Sequence for WAL IDs
CREATE SEQUENCE IF NOT EXISTS wal_id_seq;

-- Checkpoints table
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id INTEGER PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    state_version INTEGER NOT NULL,
    last_gate_seq INTEGER NOT NULL,   -- Last gate sequence number applied
    state_path VARCHAR NOT NULL,      -- Path to state Parquet files
    checksum VARCHAR,                 -- SHA256 of state data (optional)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, state_version)
);

-- Sequence for checkpoint IDs
CREATE SEQUENCE IF NOT EXISTS checkpoint_id_seq;

-- Index for efficient queries
CREATE INDEX IF NOT EXISTS idx_wal_run_status ON wal(run_id, status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id, state_version);
