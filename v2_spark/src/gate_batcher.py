"""
Gate batching for efficient Spark execution.

Groups consecutive gates into batches to reduce the number of
Spark jobs and checkpointing overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterator

from .gates import Gate


@dataclass
class GateBatch:
    """
    A batch of consecutive gates to be applied together.
    
    Attributes:
        batch_id: Unique identifier for this batch.
        start_seq: Starting gate sequence number (inclusive).
        end_seq: Ending gate sequence number (exclusive).
        gates: List of Gate objects in this batch.
    """
    batch_id: int
    start_seq: int
    end_seq: int
    gates: List[Gate]
    
    @property
    def size(self) -> int:
        """Number of gates in this batch."""
        return len(self.gates)
    
    def __repr__(self) -> str:
        return f"GateBatch(id={self.batch_id}, gates[{self.start_seq}:{self.end_seq}], size={self.size})"


class GateBatcher:
    """
    Groups gates into batches for efficient Spark execution.
    
    Batching reduces:
    - Number of Spark job submissions
    - Number of checkpoint operations
    - DataFrame materialization overhead
    """
    
    def __init__(self, batch_size: int = 10):
        """
        Initialize the batcher.
        
        Args:
            batch_size: Maximum number of gates per batch.
        """
        self.batch_size = batch_size
    
    def create_batches(
        self, 
        gates: List[Gate], 
        start_seq: int = 0
    ) -> List[GateBatch]:
        """
        Group gates into batches.
        
        Args:
            gates: List of gates to batch.
            start_seq: Starting sequence number (for resume from checkpoint).
            
        Returns:
            List of GateBatch objects.
        """
        batches = []
        batch_id = 0
        
        for i in range(0, len(gates), self.batch_size):
            batch_gates = gates[i:i + self.batch_size]
            batch = GateBatch(
                batch_id=batch_id,
                start_seq=start_seq + i,
                end_seq=start_seq + i + len(batch_gates),
                gates=batch_gates,
            )
            batches.append(batch)
            batch_id += 1
        
        return batches
    
    def iterate_batches(
        self, 
        gates: List[Gate], 
        start_seq: int = 0
    ) -> Iterator[GateBatch]:
        """
        Iterate over gate batches (generator version).
        
        Args:
            gates: List of gates to batch.
            start_seq: Starting sequence number.
            
        Yields:
            GateBatch objects.
        """
        batch_id = 0
        
        for i in range(0, len(gates), self.batch_size):
            batch_gates = gates[i:i + self.batch_size]
            yield GateBatch(
                batch_id=batch_id,
                start_seq=start_seq + i,
                end_seq=start_seq + i + len(batch_gates),
                gates=batch_gates,
            )
            batch_id += 1
    
    def get_batches_from_seq(
        self, 
        gates: List[Gate], 
        from_seq: int
    ) -> List[GateBatch]:
        """
        Get batches starting from a specific gate sequence number.
        
        Used for recovery: skip already-processed gates.
        
        Args:
            gates: Full list of gates.
            from_seq: Sequence number to start from (skip gates before this).
            
        Returns:
            List of GateBatch objects for remaining gates.
        """
        remaining_gates = gates[from_seq:]
        return self.create_batches(remaining_gates, start_seq=from_seq)
