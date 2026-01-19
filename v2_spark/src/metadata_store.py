"""
Metadata store using DuckDB for WAL and checkpoint management.

Provides a unified interface for metadata operations:
- WAL entries (PENDING, COMMITTED, FAILED)
- Checkpoint records
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import duckdb

from .config import SimulatorConfig


@dataclass
class WALEntry:
    """Write-ahead log entry."""
    wal_id: int
    run_id: str
    gate_start: int
    gate_end: int
    state_version_in: int
    state_version_out: int
    status: str
    created_at: datetime
    committed_at: Optional[datetime] = None


@dataclass
class CheckpointRecord:
    """Checkpoint record."""
    checkpoint_id: int
    run_id: str
    state_version: int
    last_gate_seq: int
    state_path: str
    checksum: Optional[str]
    created_at: datetime


class MetadataStore:
    """
    Metadata store backed by DuckDB.
    
    Manages:
    - Write-ahead log entries for batch operations
    - Checkpoint records for recovery
    """
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            db_path = self.config.metadata_db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(db_path))
            self._initialize_schema()
        return self._conn
    
    def _initialize_schema(self):
        """Initialize database schema."""
        schema_path = Path(__file__).parent.parent / "sql" / "metadata_schema.sql"
        if schema_path.exists():
            schema_sql = schema_path.read_text()
            self._conn.execute(schema_sql)
        else:
            # Inline schema if file not found
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS wal (
                    wal_id INTEGER PRIMARY KEY,
                    run_id VARCHAR NOT NULL,
                    gate_start INTEGER NOT NULL,
                    gate_end INTEGER NOT NULL,
                    state_version_in INTEGER NOT NULL,
                    state_version_out INTEGER NOT NULL,
                    status VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    committed_at TIMESTAMP
                );
                CREATE SEQUENCE IF NOT EXISTS wal_id_seq;
                
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY,
                    run_id VARCHAR NOT NULL,
                    state_version INTEGER NOT NULL,
                    last_gate_seq INTEGER NOT NULL,
                    state_path VARCHAR NOT NULL,
                    checksum VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, state_version)
                );
                CREATE SEQUENCE IF NOT EXISTS checkpoint_id_seq;
            """)
    
    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    # =========================================================================
    # WAL Operations
    # =========================================================================
    
    def wal_create_pending(
        self,
        run_id: str,
        gate_start: int,
        gate_end: int,
        state_version_in: int,
        state_version_out: int,
    ) -> int:
        """
        Create a PENDING WAL entry before batch execution.
        
        Returns:
            WAL entry ID.
        """
        conn = self.connect()
        result = conn.execute(
            """
            INSERT INTO wal (wal_id, run_id, gate_start, gate_end, 
                           state_version_in, state_version_out, status)
            VALUES (nextval('wal_id_seq'), ?, ?, ?, ?, ?, 'PENDING')
            RETURNING wal_id
            """,
            [run_id, gate_start, gate_end, state_version_in, state_version_out]
        ).fetchone()
        return result[0]
    
    def wal_mark_committed(self, wal_id: int):
        """Mark a WAL entry as COMMITTED after successful batch execution."""
        conn = self.connect()
        conn.execute(
            """
            UPDATE wal 
            SET status = 'COMMITTED', committed_at = CURRENT_TIMESTAMP 
            WHERE wal_id = ?
            """,
            [wal_id]
        )
    
    def wal_mark_failed(self, wal_id: int):
        """Mark a WAL entry as FAILED after batch execution failure."""
        conn = self.connect()
        conn.execute(
            "UPDATE wal SET status = 'FAILED' WHERE wal_id = ?",
            [wal_id]
        )
    
    def wal_get_pending(self, run_id: str) -> List[WALEntry]:
        """Get all PENDING WAL entries for a run (for recovery)."""
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT wal_id, run_id, gate_start, gate_end, 
                   state_version_in, state_version_out, status, 
                   created_at, committed_at
            FROM wal 
            WHERE run_id = ? AND status = 'PENDING'
            ORDER BY gate_start
            """,
            [run_id]
        ).fetchall()
        
        return [
            WALEntry(
                wal_id=r[0],
                run_id=r[1],
                gate_start=r[2],
                gate_end=r[3],
                state_version_in=r[4],
                state_version_out=r[5],
                status=r[6],
                created_at=r[7],
                committed_at=r[8],
            )
            for r in rows
        ]
    
    def wal_get_last_committed(self, run_id: str) -> Optional[WALEntry]:
        """Get the last COMMITTED WAL entry for a run."""
        conn = self.connect()
        row = conn.execute(
            """
            SELECT wal_id, run_id, gate_start, gate_end, 
                   state_version_in, state_version_out, status, 
                   created_at, committed_at
            FROM wal 
            WHERE run_id = ? AND status = 'COMMITTED'
            ORDER BY gate_end DESC
            LIMIT 1
            """,
            [run_id]
        ).fetchone()
        
        if row is None:
            return None
        
        return WALEntry(
            wal_id=row[0],
            run_id=row[1],
            gate_start=row[2],
            gate_end=row[3],
            state_version_in=row[4],
            state_version_out=row[5],
            status=row[6],
            created_at=row[7],
            committed_at=row[8],
        )
    
    # =========================================================================
    # Checkpoint Operations
    # =========================================================================
    
    def checkpoint_create(
        self,
        run_id: str,
        state_version: int,
        last_gate_seq: int,
        state_path: str,
        checksum: Optional[str] = None,
    ) -> int:
        """
        Create a checkpoint record.
        
        Returns:
            Checkpoint ID.
        """
        conn = self.connect()
        # Delete existing checkpoint for this run_id+state_version if any
        conn.execute(
            "DELETE FROM checkpoints WHERE run_id = ? AND state_version = ?",
            [run_id, state_version]
        )
        # Insert new checkpoint
        result = conn.execute(
            """
            INSERT INTO checkpoints (checkpoint_id, run_id, state_version, 
                                    last_gate_seq, state_path, checksum, created_at)
            VALUES (nextval('checkpoint_id_seq'), ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            RETURNING checkpoint_id
            """,
            [run_id, state_version, last_gate_seq, state_path, checksum]
        ).fetchone()
        return result[0]
    
    def checkpoint_get_latest(self, run_id: str) -> Optional[CheckpointRecord]:
        """Get the latest checkpoint for a run."""
        conn = self.connect()
        row = conn.execute(
            """
            SELECT checkpoint_id, run_id, state_version, last_gate_seq, 
                   state_path, checksum, created_at
            FROM checkpoints 
            WHERE run_id = ?
            ORDER BY state_version DESC
            LIMIT 1
            """,
            [run_id]
        ).fetchone()
        
        if row is None:
            return None
        
        return CheckpointRecord(
            checkpoint_id=row[0],
            run_id=row[1],
            state_version=row[2],
            last_gate_seq=row[3],
            state_path=row[4],
            checksum=row[5],
            created_at=row[6],
        )
    
    def checkpoint_get_by_version(
        self, 
        run_id: str, 
        state_version: int
    ) -> Optional[CheckpointRecord]:
        """Get a specific checkpoint by version."""
        conn = self.connect()
        row = conn.execute(
            """
            SELECT checkpoint_id, run_id, state_version, last_gate_seq, 
                   state_path, checksum, created_at
            FROM checkpoints 
            WHERE run_id = ? AND state_version = ?
            """,
            [run_id, state_version]
        ).fetchone()
        
        if row is None:
            return None
        
        return CheckpointRecord(
            checkpoint_id=row[0],
            run_id=row[1],
            state_version=row[2],
            last_gate_seq=row[3],
            state_path=row[4],
            checksum=row[5],
            created_at=row[6],
        )
    
    def checkpoint_list(self, run_id: str) -> List[CheckpointRecord]:
        """List all checkpoints for a run."""
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT checkpoint_id, run_id, state_version, last_gate_seq, 
                   state_path, checksum, created_at
            FROM checkpoints 
            WHERE run_id = ?
            ORDER BY state_version
            """,
            [run_id]
        ).fetchall()
        
        return [
            CheckpointRecord(
                checkpoint_id=r[0],
                run_id=r[1],
                state_version=r[2],
                last_gate_seq=r[3],
                state_path=r[4],
                checksum=r[5],
                created_at=r[6],
            )
            for r in rows
        ]
