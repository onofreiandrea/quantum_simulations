# Non-Local Gates — Butterfly Exchange

## Implemented

When a gate touches qubit `q >= log2(chunk_size)`, the paired amplitudes
live in different chunks. The engine handles this via **butterfly exchange**:
point-to-point swaps between partner chunks.

### Classification

Given `k = log2(chunk_size)`:

| Case | qa | qb | Partner structure |
|------|----|----|-------------------|
| 1-qubit non-local | q >= k | — | chunk pair: `c XOR (1 << (q-k))` |
| 2q: qa local, qb non-local | < k | >= k | pair on qb bit, local pairs on qa |
| 2q: qa non-local, qb local | >= k | < k | pair on qa bit, local pairs on qb |
| 2q: both non-local | >= k | >= k | quad of 4 chunks |

### Algorithm (1-qubit example)

```
partner_bit = q - k

for each chunk c where bit partner_bit = 0:
    partner = c XOR (1 << partner_bit)
    load chunk_c, chunk_partner
    
    # element-wise 2×2 update (the entire chunk is one big batch)
    new_c      = U[0,0] * chunk_c + U[0,1] * chunk_partner
    new_partner = U[1,0] * chunk_c + U[1,1] * chunk_partner
    
    write both back
```

Cost: same I/O as a local gate (1 read + 1 write per chunk), but loads
two chunks at once (2× peak memory per gate application).

### Multi-node future

On a cluster, the butterfly maps to MPI-style `sendrecv` between the
two nodes holding partner chunks. No global shuffle needed — each chunk
talks to exactly one partner.

## Chunk size tuning

Larger chunk_size → more qubits local → fewer butterfly exchanges.
Trade-off: memory per node vs communication rounds.
