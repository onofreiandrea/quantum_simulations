# Storage Specification

## Layout

```
<work_dir>/
├── state_a/
│   ├── manifest.json
│   └── chunks/
│       ├── chunk_000000.bin
│       ├── chunk_000001.bin
│       └── ...
├── state_b/
│   ├── manifest.json
│   └── chunks/
│       └── ...
├── wal.json
└── qubit_mapping.json   (only if staging produced a non-identity permutation)
```

Two buffers (`state_a`, `state_b`) alternate as source and destination
(double-buffering).  The WAL tracks which buffer holds the latest
committed state.

## Chunks

- dtype: `complex64` (8 bytes per amplitude)
- Default chunk_size: 2^20 (1M amplitudes = 8 MB)
- Each `.bin` file is a raw contiguous array of `complex64`
- Chunk `i` holds amplitudes `[i * chunk_size .. (i+1) * chunk_size)`

## Manifest

```json
{
  "n_qubits": 10,
  "chunk_size": 1048576,
  "n_chunks": 1,
  "dtype": "complex64",
  "chunks": ["chunk_000000.bin"],
  "created": 1700000000.0
}
```

Invariant: `chunk_size * n_chunks == 2^n_qubits`.

Each buffer directory contains its own `manifest.json`.  A buffer is
considered valid iff its `manifest.json` exists and passes validation.

## Atomic commit protocol

1. Write each chunk via temp file + `fsync` + `os.replace`
2. Write `manifest.json.tmp`, `fsync`, `os.replace` to `manifest.json`
3. Update `wal.json` atomically to flip `committed_buf`

The source buffer is **never modified** during a step — only the
destination buffer is written to.  On crash, recovery wipes the
destination and re-runs from the intact source.
