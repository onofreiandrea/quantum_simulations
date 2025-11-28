-- Quantum state: sparse
CREATE TABLE IF NOT EXISTS state (
    version INT,
    idx BIGINT,
    real DOUBLE,
    imag DOUBLE,
    PRIMARY KEY (version, idx)
);

-- Gate matrix elements
CREATE TABLE IF NOT EXISTS gate_matrix (
    gate_name TEXT,
    arity INT,
    row INT,
    col INT,
    real DOUBLE,
    imag DOUBLE,
    PRIMARY KEY (gate_name, arity, row, col)
);

-- Write-ahead log
CREATE TABLE IF NOT EXISTS wal (
    wal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INT,
    gate_name TEXT,
    qubits TEXT,
    params TEXT,
    status TEXT CHECK (status IN ('PENDING', 'COMMITTED'))
);

-- Checkpoints
CREATE TABLE IF NOT EXISTS checkpoint (
    version INT PRIMARY KEY,
    path TEXT,
    checksum TEXT,
    timestamp TEXT
);

