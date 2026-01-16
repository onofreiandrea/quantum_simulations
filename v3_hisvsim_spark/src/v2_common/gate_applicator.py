"""
Gate application using Spark DataFrame operations.

Implements quantum gate application as DataFrame transformations:
- Broadcast join with gate matrices
- Bitwise operations for index manipulation
- Complex number arithmetic
- GroupBy aggregation for amplitude summation
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import pyspark.sql.functions as F
from pyspark.broadcast import Broadcast

import numpy as np

from v2_common.gates import Gate


# Schema for gate matrix DataFrame
GATE_MATRIX_SCHEMA = StructType([
    StructField("gate_name", StringType(), nullable=False),
    StructField("arity", IntegerType(), nullable=False),
    StructField("row", IntegerType(), nullable=False),
    StructField("col", IntegerType(), nullable=False),
    StructField("real", DoubleType(), nullable=False),
    StructField("imag", DoubleType(), nullable=False),
])


# Helper functions for Spark bitwise operations
# Using SQL expressions for PySpark 4.0+ compatibility
# IMPORTANT: Must use BIGINT (64-bit) for qubit indices > 31

def spark_shift_right(col, n: int):
    """Shift column right by n bits: col >> n (64-bit safe)"""
    # Cast to long to ensure 64-bit arithmetic
    return F.shiftright(col.cast("long"), n)


def spark_shift_left(col, n: int):
    """Shift column left by n bits: col << n (64-bit safe)"""
    # Cast to long to ensure 64-bit arithmetic for large qubit indices
    return F.shiftleft(col.cast("long"), n)


def spark_shift_left_literal(n: int) -> int:
    """Return 1 << n as a Python int (for use as literal in expressions)."""
    return 1 << n


def spark_bitwise_and(col1, col2):
    """Bitwise AND using SQL expression: col1 & col2"""
    # Ensure 64-bit arithmetic
    return col1.cast("long").bitwiseAND(col2)


def spark_bitwise_or(col1, col2):
    """Bitwise OR using SQL expression: col1 | col2"""
    return col1.cast("long").bitwiseOR(col2)


def spark_bitwise_xor(col1, col2):
    """Bitwise XOR using SQL expression: col1 ^ col2"""
    return col1.cast("long").bitwiseXOR(col2)


class GateApplicator:
    """
    Applies quantum gates to state vectors using Spark DataFrame operations.
    
    Gate matrices are broadcasted to all executors for efficient joins.
    State transformations use:
    - Bitwise operations for qubit index extraction/insertion
    - Complex number multiplication
    - GroupBy aggregation for amplitude summation
    
    For distributed execution, the state is repartitioned after each gate
    to ensure parallelism across the cluster.
    """
    
    def __init__(self, spark: SparkSession, num_partitions: int = 16):
        """
        Initialize the gate applicator.
        
        Args:
            spark: SparkSession instance.
            num_partitions: Number of partitions for distributed state.
                           Set higher for larger clusters.
        """
        self.spark = spark
        self.num_partitions = num_partitions
        self._gate_matrices: Dict[str, DataFrame] = {}
        self._gate_matrix_df: DataFrame | None = None
    
    def register_gates(self, gates: List[Gate]):
        """
        Register gate matrices for all unique gates in the circuit.
        
        Creates a single DataFrame containing all gate matrices,
        which will be broadcast-joined during gate application.
        
        Args:
            gates: List of Gate objects to register.
        """
        seen: set[Tuple[str, int]] = set()
        matrix_rows = []
        
        for gate in gates:
            arity = 2 if gate.two_qubit_gate else 1
            key = (gate.gate_name, arity)
            
            if key in seen:
                continue
            seen.add(key)
            
            if gate.two_qubit_gate:
                # 2-qubit gate: 4x4 matrix
                tensor = gate.tensor.reshape(4, 4)
                for row in range(4):
                    for col in range(4):
                        val = tensor[row, col]
                        matrix_rows.append((
                            gate.gate_name,
                            2,
                            row,
                            col,
                            float(np.real(val)),
                            float(np.imag(val)),
                        ))
            else:
                # 1-qubit gate: 2x2 matrix
                tensor = gate.tensor
                for row in range(2):
                    for col in range(2):
                        val = tensor[row, col]
                        matrix_rows.append((
                            gate.gate_name,
                            1,
                            row,
                            col,
                            float(np.real(val)),
                            float(np.imag(val)),
                        ))
        
        self._gate_matrix_df = self.spark.createDataFrame(
            matrix_rows, 
            schema=GATE_MATRIX_SCHEMA
        ).cache()  # Cache for repeated use
    
    def apply_one_qubit_gate(
        self, 
        state_df: DataFrame, 
        gate_name: str, 
        qubit: int
    ) -> DataFrame:
        """
        Apply a 1-qubit gate to the state vector.
        
        The transformation:
        1. Extract qubit bit from each basis state index
        2. Join with gate matrix on (gate_name, col=qubit_bit)
        3. Compute new index by replacing qubit bit with row from matrix
        4. Complex multiply state amplitude with matrix element
        5. Group by new index and sum amplitudes
        
        Args:
            state_df: Current state DataFrame (idx, real, imag).
            gate_name: Name of the gate in gate_matrix.
            qubit: Target qubit index.
            
        Returns:
            New state DataFrame after gate application.
        """
        if self._gate_matrix_df is None:
            raise RuntimeError("Gate matrices not registered. Call register_gates first.")
        
        # Filter gate matrix for this specific gate
        gate_matrix = self._gate_matrix_df.filter(
            (F.col("gate_name") == gate_name) & (F.col("arity") == 1)
        )
        
        # Broadcast the gate matrix (small data)
        gate_matrix_bc = F.broadcast(gate_matrix)
        
        # Pre-compute bit mask
        bit_mask = 1 << qubit
        
        # Apply the gate transformation using Spark bitwise functions
        result = (
            state_df
            # Extract the qubit bit from index: (idx >> qubit) & 1
            .withColumn("qubit_bit", 
                spark_bitwise_and(spark_shift_right(F.col("idx"), qubit), F.lit(1)))
            # OPTIMIZATION: Join with broadcast gate matrix on col = qubit_bit
            # gate_matrix_bc is already broadcast, so join will use broadcast join
            .join(
                gate_matrix_bc.select(
                    F.col("row").alias("g_row"),
                    F.col("col").alias("g_col"),
                    F.col("real").alias("g_real"),
                    F.col("imag").alias("g_imag"),
                ),
                F.col("qubit_bit") == F.col("g_col"),
                "inner"
            )
            # Compute new index: XOR to clear qubit bit, then OR to set new value
            # new_idx = (idx ^ (qubit_bit << qubit)) | (g_row << qubit)
            .withColumn(
                "new_idx",
                spark_bitwise_or(
                    spark_bitwise_xor(
                        F.col("idx"),
                        spark_shift_left(F.col("qubit_bit"), qubit)
                    ),
                    spark_shift_left(F.col("g_row"), qubit)
                )
            )
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            .withColumn(
                "new_real",
                F.col("g_real") * F.col("real") - F.col("g_imag") * F.col("imag")
            )
            .withColumn(
                "new_imag",
                F.col("g_real") * F.col("imag") + F.col("g_imag") * F.col("real")
            )
            # Group by new index and sum amplitudes
            .groupBy("new_idx")
            .agg(
                F.sum("new_real").alias("real"),
                F.sum("new_imag").alias("imag")
            )
            # Rename and select final columns
            .select(
                F.col("new_idx").alias("idx"),
                F.col("real"),
                F.col("imag")
            )
            # Filter out zero amplitudes (sparse representation)
            .filter(
                (F.abs(F.col("real")) > 1e-15) | (F.abs(F.col("imag")) > 1e-15)
            )
        )
        
        # Repartition for distributed execution
        # This ensures the state is spread across workers for parallelism
        if self.num_partitions > 1:
            result = result.repartition(self.num_partitions, "idx")
        
        return result
    
    def apply_two_qubit_gate(
        self, 
        state_df: DataFrame, 
        gate_name: str, 
        qubit0: int, 
        qubit1: int
    ) -> DataFrame:
        """
        Apply a 2-qubit gate to the state vector.
        
        Similar to 1-qubit gate but handles 2 qubits:
        - col is 2-bit value from (qubit1_bit << 1) | qubit0_bit
        - row is 2-bit value that replaces both qubit bits
        
        Args:
            state_df: Current state DataFrame (idx, real, imag).
            gate_name: Name of the gate in gate_matrix.
            qubit0: First qubit index.
            qubit1: Second qubit index.
            
        Returns:
            New state DataFrame after gate application.
        """
        if self._gate_matrix_df is None:
            raise RuntimeError("Gate matrices not registered. Call register_gates first.")
        
        # Filter gate matrix for this specific gate
        gate_matrix = self._gate_matrix_df.filter(
            (F.col("gate_name") == gate_name) & (F.col("arity") == 2)
        )
        
        # OPTIMIZATION: Always broadcast gate matrices (they're small)
        # Use explicit broadcast hint for better Spark optimization
        gate_matrix_bc = F.broadcast(gate_matrix)
        
        # Apply the gate transformation using Spark bitwise functions
        result = (
            state_df
            # Extract individual qubit bits: (idx >> qubit) & 1
            .withColumn("bit0", 
                spark_bitwise_and(spark_shift_right(F.col("idx"), qubit0), F.lit(1)))
            .withColumn("bit1", 
                spark_bitwise_and(spark_shift_right(F.col("idx"), qubit1), F.lit(1)))
            # Combine into 2-bit column value: bit0 | (bit1 << 1)
            .withColumn(
                "qubit_bits",
                spark_bitwise_or(F.col("bit0"), spark_shift_left(F.col("bit1"), 1))
            )
            # OPTIMIZATION: Join with broadcast gate matrix on col = qubit_bits
            # gate_matrix_bc is already broadcast, so join will use broadcast join
            .join(
                gate_matrix_bc.select(
                    F.col("row").alias("g_row"),
                    F.col("col").alias("g_col"),
                    F.col("real").alias("g_real"),
                    F.col("imag").alias("g_imag"),
                ),
                F.col("qubit_bits") == F.col("g_col"),
                "inner"
            )
            # Compute new index using XOR to clear bits, OR to set new values
            # Step 1: Clear both bits: idx ^ (bit0 << qubit0) ^ (bit1 << qubit1)
            .withColumn(
                "cleared_idx",
                spark_bitwise_xor(
                    spark_bitwise_xor(
                        F.col("idx"),
                        spark_shift_left(F.col("bit0"), qubit0)
                    ),
                    spark_shift_left(F.col("bit1"), qubit1)
                )
            )
            # Step 2: Extract new bits from g_row
            .withColumn("new_bit0", spark_bitwise_and(F.col("g_row"), F.lit(1)))
            .withColumn("new_bit1", 
                spark_bitwise_and(spark_shift_right(F.col("g_row"), 1), F.lit(1)))
            # Step 3: Set new bits: cleared | (new_bit0 << qubit0) | (new_bit1 << qubit1)
            .withColumn(
                "new_idx",
                spark_bitwise_or(
                    spark_bitwise_or(
                        F.col("cleared_idx"),
                        spark_shift_left(F.col("new_bit0"), qubit0)
                    ),
                    spark_shift_left(F.col("new_bit1"), qubit1)
                )
            )
            # Complex multiplication
            .withColumn(
                "new_real",
                F.col("g_real") * F.col("real") - F.col("g_imag") * F.col("imag")
            )
            .withColumn(
                "new_imag",
                F.col("g_real") * F.col("imag") + F.col("g_imag") * F.col("real")
            )
            # Group by new index and sum amplitudes
            .groupBy("new_idx")
            .agg(
                F.sum("new_real").alias("real"),
                F.sum("new_imag").alias("imag")
            )
            # Rename and select final columns
            .select(
                F.col("new_idx").alias("idx"),
                F.col("real"),
                F.col("imag")
            )
            # Filter out zero amplitudes
            .filter(
                (F.abs(F.col("real")) > 1e-15) | (F.abs(F.col("imag")) > 1e-15)
            )
        )
        
        # Repartition for distributed execution
        if self.num_partitions > 1:
            result = result.repartition(self.num_partitions, "idx")
        
        return result
    
    def apply_gate(self, state_df: DataFrame, gate: Gate) -> DataFrame:
        """
        Apply a gate to the state vector.
        
        Args:
            state_df: Current state DataFrame.
            gate: Gate object to apply.
            
        Returns:
            New state DataFrame after gate application.
        """
        if gate.two_qubit_gate:
            q0, q1 = gate.qubits
            return self.apply_two_qubit_gate(state_df, gate.gate_name, q0, q1)
        else:
            (q,) = gate.qubits
            return self.apply_one_qubit_gate(state_df, gate.gate_name, q)
    
    def apply_gates(self, state_df: DataFrame, gates: List[Gate]) -> DataFrame:
        """
        Apply a sequence of gates to the state vector.
        
        Builds a lazy DataFrame plan for all gates, which is executed
        when an action is triggered (e.g., save to Parquet).
        
        Args:
            state_df: Current state DataFrame.
            gates: List of gates to apply in order.
            
        Returns:
            New state DataFrame after all gates are applied.
        """
        current_state = state_df
        for gate in gates:
            current_state = self.apply_gate(current_state, gate)
        return current_state
    
    def cleanup(self):
        """Release cached resources."""
        if self._gate_matrix_df is not None:
            self._gate_matrix_df.unpersist()
            self._gate_matrix_df = None
