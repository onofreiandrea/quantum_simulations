"""
State merging for HiSVSIM + Spark integration.

Merges states from parallel partition simulations.

Key insight:
- Independent partitions (no qubit overlap): Tensor product
- Overlapping partitions: Match shared qubits, then combine
"""
from __future__ import annotations

from typing import List, Set, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
import numpy as np

from .driver import PartitionResult
from .v2_common import config

SimulatorConfig = config.SimulatorConfig


def spark_bitwise_and(col1, col2):
    """Spark bitwise AND operation."""
    return col1.cast("long").bitwiseAND(col2)


def spark_bitwise_or(col1, col2):
    """Spark bitwise OR operation."""
    return col1.cast("long").bitwiseOR(col2)


def spark_shift_left(col, shift):
    """Spark left shift operation."""
    return F.expr(f"CAST({col} AS BIGINT) << {shift}")


def spark_shift_right(col, shift):
    """Spark right shift operation."""
    return F.expr(f"CAST({col} AS BIGINT) >> {shift}")


class StateMerger:
    """
    Merge states from parallel partition simulations.
    
    Strategy:
    1. If partitions are independent (no qubit overlap): Tensor product
    2. If partitions overlap: Match shared qubits, combine amplitudes
    """
    
    def __init__(self, spark: SparkSession, config: SimulatorConfig):
        """
        Initialize state merger.
        
        Args:
            spark: SparkSession instance.
            config: Simulator configuration.
        """
        self.spark = spark
        self.config = config
    
    def merge_partitions(
        self, 
        partition_results: List[PartitionResult], 
        n_qubits: int
    ) -> DataFrame:
        """
        Merge states from multiple partitions.
        
        NOTE: This method is kept for future parallel execution.
        Currently, partitions are simulated sequentially in the driver,
        so merging is not needed. This is a placeholder for when we
        implement true parallel partition execution.
        
        Args:
            partition_results: List of PartitionResult objects.
            n_qubits: Total number of qubits.
            
        Returns:
            Merged state DataFrame.
        """
        # For now, if we have partition results, just return the last one
        # (since they're simulated sequentially)
        if partition_results:
            return partition_results[-1].state_df
        
        # Fallback: return empty state
        return self.spark.createDataFrame(
            [(0, 1.0, 0.0)],
            schema=["idx", "real", "imag"]
        )
    
    def _check_overlaps(
        self, 
        partition_results: List[PartitionResult]
    ) -> bool:
        """Check if partitions have qubit overlaps."""
        for i, result_i in enumerate(partition_results):
            for j, result_j in enumerate(partition_results):
                if i < j:
                    if result_i.qubits_used & result_j.qubits_used:
                        return True
        return False
    
    def _merge_independent_partitions(
        self, 
        partition_results: List[PartitionResult], 
        n_qubits: int
    ) -> DataFrame:
        """
        Merge independent partitions.
        
        Key insight: Each partition operates on the FULL n_qubits state space,
        but only affects its own qubits. Since partitions are independent
        (no qubit overlap), we can combine them by:
        
        1. If partitions truly don't overlap: They should produce the same
           result as sequential execution, so we can just take the final
           partition's state (since earlier partitions only affect their qubits)
        
        2. Actually, if partitions are independent, we need to apply them
           sequentially, not in parallel! So merging should be sequential.
        
        For now, we'll use sequential application: each partition's gates
        are applied to the state from the previous partition.
        """
        # Start with initial state |00...0⟩
        merged = self.spark.createDataFrame(
            [(0, 1.0, 0.0)],
            schema=["idx", "real", "imag"]
        )
        
        # Sequentially apply each partition's state
        # This is a simplified approach - ideally partitions would be
        # truly independent and we could combine them differently
        for result in partition_results:
            # For independent partitions, we should be able to combine
            # but since both operate on full space, we need to be careful
            
            # Simplified: union and group (assumes sequential execution)
            # This works if partitions are applied sequentially
            merged = (
                merged.union(result.state_df)
                .groupBy("idx")
                .agg(
                    F.sum("real").alias("real"),
                    F.sum("imag").alias("imag")
                )
                .filter(
                    (F.abs(F.col("real")) > 1e-15) | 
                    (F.abs(F.col("imag")) > 1e-15)
                )
            )
        
        return merged
    
    def _merge_overlapping_partitions(
        self, 
        partition_results: List[PartitionResult], 
        n_qubits: int
    ) -> DataFrame:
        """
        Merge partitions that share qubits.
        
        Strategy:
        1. Sort partitions by size (largest first)
        2. Start with largest partition
        3. For each subsequent partition:
           - Find shared qubits
           - Match states where shared qubits have same values
           - Combine amplitudes properly
        
        This is more complex and requires careful qubit matching.
        """
        # Sort partitions by size (largest first)
        sorted_results = sorted(
            partition_results,
            key=lambda r: len(r.qubits_used),
            reverse=True
        )
        
        # Start with largest partition
        merged = sorted_results[0].state_df
        merged_qubits = sorted_results[0].qubits_used
        
        # Sequentially merge others
        for result in sorted_results[1:]:
            shared_qubits = merged_qubits & result.qubits_used
            
            if not shared_qubits:
                # No overlap: use tensor product
                merged = self._tensor_product_two_states(
                    merged, merged_qubits,
                    result.state_df, result.qubits_used
                )
            else:
                # Has overlap: match shared qubits
                merged = self._merge_with_shared_qubits(
                    merged, merged_qubits,
                    result.state_df, result.qubits_used,
                    shared_qubits, n_qubits
                )
            
            merged_qubits = merged_qubits | result.qubits_used
        
        return merged
    
    def _tensor_product_two_states(
        self,
        state1_df: DataFrame,
        qubits1: Set[int],
        state2_df: DataFrame,
        qubits2: Set[int]
    ) -> DataFrame:
        """Compute tensor product of two independent states."""
        # Find qubit offset (number of qubits in state1)
        qubit_offset = len(qubits1)
        
        return (
            state1_df.alias("state1")
            .crossJoin(state2_df.alias("state2"))
            .withColumn(
                "final_idx",
                spark_bitwise_or(
                    F.col("state1.idx"),
                    spark_shift_left(F.col("state2.idx"), qubit_offset)
                )
            )
            .withColumn(
                "final_real",
                F.col("state1.real") * F.col("state2.real") - 
                F.col("state1.imag") * F.col("state2.imag")
            )
            .withColumn(
                "final_imag",
                F.col("state1.real") * F.col("state2.imag") + 
                F.col("state1.imag") * F.col("state2.real")
            )
            .select(
                F.col("final_idx").alias("idx"),
                F.col("final_real").alias("real"),
                F.col("final_imag").alias("imag")
            )
            .filter(
                (F.abs(F.col("real")) > 1e-15) | 
                (F.abs(F.col("imag")) > 1e-15)
            )
        )
    
    def _merge_with_shared_qubits(
        self,
        state1_df: DataFrame,
        qubits1: Set[int],
        state2_df: DataFrame,
        qubits2: Set[int],
        shared_qubits: Set[int],
        n_qubits: int
    ) -> DataFrame:
        """
        Merge two states that share qubits.
        
        Strategy:
        - Extract shared qubit values from both states
        - Match states where shared qubits agree
        - Combine amplitudes for matching states
        - Properly handle qubit ordering
        """
        # For now, use a simplified approach:
        # If partitions overlap, they should have been simulated sequentially
        # So we just need to combine amplitudes where indices match
        
        # This is a placeholder - full implementation would:
        # 1. Extract shared qubit bits from idx1 and idx2
        # 2. Match where shared qubits agree
        # 3. Combine non-shared qubits properly
        
        # Simplified: union and group by index (assumes sequential execution)
        merged = (
            state1_df.union(state2_df)
            .groupBy("idx")
            .agg(
                F.sum("real").alias("real"),
                F.sum("imag").alias("imag")
            )
            .filter(
                (F.abs(F.col("real")) > 1e-15) | 
                (F.abs(F.col("imag")) > 1e-15)
            )
        )
        
        return merged
