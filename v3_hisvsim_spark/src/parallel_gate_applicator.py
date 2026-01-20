"""
PARALLEL Gate Applicator - True parallel gate execution.

When gates operate on INDEPENDENT qubits (no overlap), they can be applied
simultaneously in a single DataFrame transformation.

Example:
  H(0) and H(1) are independent → apply BOTH in one pass
  H(0) and CNOT(0,1) share qubit 0 → must be sequential

This is TRUE parallel gate execution, not just state distribution.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Set
from itertools import product

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, LongType
import pyspark.sql.functions as F

import numpy as np

from v2_common.gates import Gate


def spark_shift_right(col, n: int):
    """Shift column right by n bits."""
    return F.shiftright(col.cast("long"), n)


def spark_shift_left(col, n: int):
    """Shift column left by n bits."""
    return F.shiftleft(col.cast("long"), n)


def spark_bitwise_and(col1, col2):
    """Bitwise AND."""
    return col1.cast("long").bitwiseAND(col2)


def spark_bitwise_or(col1, col2):
    """Bitwise OR."""
    return col1.cast("long").bitwiseOR(col2)


def spark_bitwise_xor(col1, col2):
    """Bitwise XOR."""
    return col1.cast("long").bitwiseXOR(col2)


class ParallelGateApplicator:
    """
    Applies quantum gates with TRUE parallel execution.
    
    Key innovation: Independent gates are fused into a single transformation.
    
    Example with H(0), H(1), H(2) (all independent):
    - Old approach: 3 sequential transformations
    - New approach: 1 combined transformation with tensor product
    """
    
    def __init__(self, spark: SparkSession, num_partitions: int = 16):
        self.spark = spark
        self.num_partitions = num_partitions
        self._gate_cache: Dict[str, np.ndarray] = {}
    
    def register_gates(self, gates: List[Gate]):
        """Cache gate tensors for reuse."""
        for gate in gates:
            key = gate.gate_name
            if key not in self._gate_cache:
                self._gate_cache[key] = gate.tensor.copy()
    
    def apply_gates_parallel(
        self, 
        state_df: DataFrame, 
        gates: List[Gate]
    ) -> DataFrame:
        """
        Apply multiple INDEPENDENT gates in parallel.
        
        REQUIREMENT: All gates must operate on different qubits!
        
        This creates a combined tensor product transformation:
        H(0) ⊗ H(1) ⊗ I(2) is applied as a single operation.
        
        NOTE: For safety, only single-qubit gates are parallelized.
        Two-qubit gates are applied sequentially to avoid complex tensor issues.
        """
        if len(gates) == 0:
            return state_df
        
        if len(gates) == 1:
            return self._apply_single_gate(state_df, gates[0])
        
        # Separate single-qubit and two-qubit gates
        single_qubit_gates = [g for g in gates if not g.two_qubit_gate]
        two_qubit_gates = [g for g in gates if g.two_qubit_gate]
        
        current_state = state_df
        
        # Apply single-qubit gates in parallel (the main optimization!)
        if len(single_qubit_gates) > 1:
            # Verify single-qubit gates are independent
            all_qubits: Set[int] = set()
            for gate in single_qubit_gates:
                gate_qubits = set(gate.qubits)
                if gate_qubits & all_qubits:
                    # If overlap, apply sequentially
                    for g in single_qubit_gates:
                        current_state = self._apply_single_gate(current_state, g)
                    break
                all_qubits.update(gate_qubits)
            else:
                # All independent - build combined transformation
                combined_matrix = self._build_combined_matrix(single_qubit_gates)
                current_state = self._apply_combined_matrix(current_state, combined_matrix, single_qubit_gates)
        elif len(single_qubit_gates) == 1:
            current_state = self._apply_single_gate(current_state, single_qubit_gates[0])
        
        # Apply two-qubit gates sequentially (safer)
        for gate in two_qubit_gates:
            current_state = self._apply_single_gate(current_state, gate)
        
        return current_state
    
    def _build_combined_matrix(self, gates: List[Gate]) -> List[Tuple]:
        """
        Build the combined transformation matrix for parallel gates.
        
        For gates on qubits [q0, q1, ...], we build all (input_bits -> output_bits) mappings
        with their complex coefficients.
        
        Returns: List of (input_pattern, output_pattern, real, imag)
        """
        # Collect all qubits and their gate matrices
        qubit_to_matrix: Dict[int, np.ndarray] = {}
        
        for gate in gates:
            if gate.two_qubit_gate:
                # 2-qubit gate - more complex handling
                q0, q1 = gate.qubits
                matrix = gate.tensor.reshape(4, 4)
                # Store as combined entry
                qubit_to_matrix[(q0, q1)] = matrix
            else:
                q = gate.qubits[0]
                qubit_to_matrix[q] = gate.tensor
        
        # Build transformation entries
        entries = []
        
        # Get sorted list of single qubits and qubit pairs
        single_qubits = sorted([k for k in qubit_to_matrix.keys() if isinstance(k, int)])
        qubit_pairs = [k for k in qubit_to_matrix.keys() if isinstance(k, tuple)]
        
        # For single-qubit gates, compute tensor product
        if single_qubits and not qubit_pairs:
            entries = self._tensor_product_single_qubits(single_qubits, qubit_to_matrix)
        elif qubit_pairs and not single_qubits:
            entries = self._tensor_product_qubit_pairs(qubit_pairs, qubit_to_matrix)
        else:
            # Mixed - handle each gate separately but in one pass
            entries = self._build_mixed_entries(gates)
        
        return entries
    
    def _tensor_product_single_qubits(
        self, 
        qubits: List[int], 
        qubit_to_matrix: Dict
    ) -> List[Tuple]:
        """
        Compute tensor product for single-qubit gates.
        
        For H(0) ⊗ H(1), we enumerate all 2^n input patterns and compute outputs.
        """
        n = len(qubits)
        entries = []
        
        # Enumerate all input bit patterns
        for input_bits in range(2**n):
            # For each input pattern, compute all output contributions
            for output_bits in range(2**n):
                # Compute the combined matrix element
                coeff = complex(1.0, 0.0)
                
                for i, q in enumerate(qubits):
                    in_bit = (input_bits >> i) & 1
                    out_bit = (output_bits >> i) & 1
                    matrix = qubit_to_matrix[q]
                    coeff *= matrix[out_bit, in_bit]
                
                if abs(coeff) > 1e-15:
                    entries.append((
                        qubits,  # Which qubits
                        input_bits, 
                        output_bits, 
                        float(coeff.real), 
                        float(coeff.imag)
                    ))
        
        return entries
    
    def _tensor_product_qubit_pairs(
        self, 
        qubit_pairs: List[Tuple], 
        qubit_to_matrix: Dict
    ) -> List[Tuple]:
        """Handle 2-qubit gates in parallel."""
        entries = []
        
        for pair in qubit_pairs:
            q0, q1 = pair
            matrix = qubit_to_matrix[pair]
            
            for in_bits in range(4):
                for out_bits in range(4):
                    coeff = matrix[out_bits, in_bits]
                    if abs(coeff) > 1e-15:
                        entries.append((
                            [q0, q1],
                            in_bits,
                            out_bits,
                            float(coeff.real),
                            float(coeff.imag)
                        ))
        
        return entries
    
    def _build_mixed_entries(self, gates: List[Gate]) -> List[Tuple]:
        """Build entries for mixed single/two-qubit gates."""
        # For simplicity, we'll handle this by creating a combined tensor
        # This is more complex but handles all cases
        
        all_qubits = []
        for gate in gates:
            all_qubits.extend(gate.qubits)
        all_qubits = sorted(set(all_qubits))
        
        n_qubits = len(all_qubits)
        qubit_to_idx = {q: i for i, q in enumerate(all_qubits)}
        
        # Build combined matrix
        size = 2 ** n_qubits
        combined = np.eye(size, dtype=complex)
        
        for gate in gates:
            gate_matrix = self._get_full_matrix(gate, all_qubits, qubit_to_idx)
            combined = gate_matrix @ combined
        
        entries = []
        for in_bits in range(size):
            for out_bits in range(size):
                coeff = combined[out_bits, in_bits]
                if abs(coeff) > 1e-15:
                    entries.append((
                        all_qubits,
                        in_bits,
                        out_bits,
                        float(coeff.real),
                        float(coeff.imag)
                    ))
        
        return entries
    
    def _get_full_matrix(
        self, 
        gate: Gate, 
        all_qubits: List[int], 
        qubit_to_idx: Dict[int, int]
    ) -> np.ndarray:
        """Embed gate matrix into full space of all_qubits."""
        n = len(all_qubits)
        size = 2 ** n
        
        if gate.two_qubit_gate:
            q0, q1 = gate.qubits
            idx0, idx1 = qubit_to_idx[q0], qubit_to_idx[q1]
            gate_matrix = gate.tensor.reshape(4, 4)
        else:
            q = gate.qubits[0]
            idx = qubit_to_idx[q]
            gate_matrix = gate.tensor
        
        # Build full matrix
        full = np.zeros((size, size), dtype=complex)
        
        for in_state in range(size):
            for out_state in range(size):
                # Check if non-gate qubits match
                match = True
                for i, q in enumerate(all_qubits):
                    if q not in gate.qubits:
                        if ((in_state >> i) & 1) != ((out_state >> i) & 1):
                            match = False
                            break
                
                if not match:
                    continue
                
                # Extract gate qubit bits
                if gate.two_qubit_gate:
                    in_bits = ((in_state >> idx0) & 1) | (((in_state >> idx1) & 1) << 1)
                    out_bits = ((out_state >> idx0) & 1) | (((out_state >> idx1) & 1) << 1)
                    full[out_state, in_state] = gate_matrix[out_bits, in_bits]
                else:
                    in_bit = (in_state >> idx) & 1
                    out_bit = (out_state >> idx) & 1
                    full[out_state, in_state] = gate_matrix[out_bit, in_bit]
        
        return full
    
    def _apply_combined_matrix(
        self, 
        state_df: DataFrame, 
        entries: List[Tuple],
        gates: List[Gate]
    ) -> DataFrame:
        """Apply the combined transformation matrix."""
        if not entries:
            return state_df
        
        # Get all qubits involved
        qubits = entries[0][0]
        if isinstance(qubits, int):
            qubits = [qubits]
        qubits = list(qubits)
        
        # Create transformation DataFrame
        schema = StructType([
            StructField("in_pattern", IntegerType(), False),
            StructField("out_pattern", IntegerType(), False),
            StructField("m_real", DoubleType(), False),
            StructField("m_imag", DoubleType(), False),
        ])
        
        transform_data = [(e[1], e[2], e[3], e[4]) for e in entries]
        transform_df = self.spark.createDataFrame(transform_data, schema)
        transform_df = F.broadcast(transform_df)
        
        # Build input pattern extraction expression
        # For qubits [0, 1, 2], extract bits and combine
        input_pattern_expr = F.lit(0)
        for i, q in enumerate(qubits):
            bit_expr = spark_bitwise_and(spark_shift_right(F.col("idx"), q), F.lit(1))
            input_pattern_expr = spark_bitwise_or(
                input_pattern_expr,
                spark_shift_left(bit_expr, i)
            )
        
        # Apply transformation
        result = (
            state_df
            .withColumn("in_pattern", input_pattern_expr.cast("int"))
            .join(transform_df, "in_pattern", "inner")
            # Compute new index: clear affected bits, set new pattern
            .withColumn("cleared_idx", self._clear_bits_expr(F.col("idx"), qubits))
            .withColumn("new_idx", self._set_bits_expr(F.col("cleared_idx"), qubits, F.col("out_pattern")))
            # Complex multiplication
            .withColumn("new_real", 
                F.col("m_real") * F.col("real") - F.col("m_imag") * F.col("imag"))
            .withColumn("new_imag",
                F.col("m_real") * F.col("imag") + F.col("m_imag") * F.col("real"))
            # Group and sum
            .groupBy("new_idx")
            .agg(
                F.sum("new_real").alias("real"),
                F.sum("new_imag").alias("imag")
            )
            .select(
                F.col("new_idx").alias("idx"),
                F.col("real"),
                F.col("imag")
            )
            .filter(
                (F.abs(F.col("real")) > 1e-15) | (F.abs(F.col("imag")) > 1e-15)
            )
        )
        
        if self.num_partitions > 1:
            result = result.repartition(self.num_partitions, "idx")
        
        return result
    
    def _clear_bits_expr(self, idx_col, qubits: List[int]):
        """Create expression to clear specific qubit bits."""
        result = idx_col
        for q in qubits:
            # Clear bit q: idx & ~(1 << q)
            mask = ~(1 << q)
            result = spark_bitwise_and(result, F.lit(mask))
        return result
    
    def _set_bits_expr(self, idx_col, qubits: List[int], pattern_col):
        """Create expression to set specific qubit bits from pattern."""
        result = idx_col
        for i, q in enumerate(qubits):
            # Extract bit i from pattern and shift to position q
            bit = spark_bitwise_and(spark_shift_right(pattern_col, i), F.lit(1))
            result = spark_bitwise_or(result, spark_shift_left(bit, q))
        return result
    
    def _apply_single_gate(self, state_df: DataFrame, gate: Gate) -> DataFrame:
        """Apply a single gate (fallback for single gates)."""
        if gate.two_qubit_gate:
            return self._apply_two_qubit_gate(state_df, gate)
        else:
            return self._apply_one_qubit_gate(state_df, gate)
    
    def _apply_one_qubit_gate(self, state_df: DataFrame, gate: Gate) -> DataFrame:
        """Apply single 1-qubit gate."""
        q = gate.qubits[0]
        matrix = gate.tensor
        
        # Create gate matrix DataFrame
        rows = []
        for row in range(2):
            for col in range(2):
                val = matrix[row, col]
                if abs(val) > 1e-15:
                    rows.append((col, row, float(val.real), float(val.imag)))
        
        schema = StructType([
            StructField("g_col", IntegerType(), False),
            StructField("g_row", IntegerType(), False),
            StructField("g_real", DoubleType(), False),
            StructField("g_imag", DoubleType(), False),
        ])
        gate_df = F.broadcast(self.spark.createDataFrame(rows, schema))
        
        result = (
            state_df
            .withColumn("qubit_bit", 
                spark_bitwise_and(spark_shift_right(F.col("idx"), q), F.lit(1)).cast("int"))
            .join(gate_df, F.col("qubit_bit") == F.col("g_col"), "inner")
            .withColumn("new_idx",
                spark_bitwise_or(
                    spark_bitwise_xor(F.col("idx"), spark_shift_left(F.col("qubit_bit"), q)),
                    spark_shift_left(F.col("g_row"), q)
                ))
            .withColumn("new_real",
                F.col("g_real") * F.col("real") - F.col("g_imag") * F.col("imag"))
            .withColumn("new_imag",
                F.col("g_real") * F.col("imag") + F.col("g_imag") * F.col("real"))
            .groupBy("new_idx")
            .agg(F.sum("new_real").alias("real"), F.sum("new_imag").alias("imag"))
            .select(F.col("new_idx").alias("idx"), "real", "imag")
            .filter((F.abs(F.col("real")) > 1e-15) | (F.abs(F.col("imag")) > 1e-15))
        )
        
        if self.num_partitions > 1:
            result = result.repartition(self.num_partitions, "idx")
        
        return result
    
    def _apply_two_qubit_gate(self, state_df: DataFrame, gate: Gate) -> DataFrame:
        """Apply single 2-qubit gate."""
        q0, q1 = gate.qubits
        matrix = gate.tensor.reshape(4, 4)
        
        rows = []
        for row in range(4):
            for col in range(4):
                val = matrix[row, col]
                if abs(val) > 1e-15:
                    rows.append((col, row, float(val.real), float(val.imag)))
        
        schema = StructType([
            StructField("g_col", IntegerType(), False),
            StructField("g_row", IntegerType(), False),
            StructField("g_real", DoubleType(), False),
            StructField("g_imag", DoubleType(), False),
        ])
        gate_df = F.broadcast(self.spark.createDataFrame(rows, schema))
        
        result = (
            state_df
            .withColumn("bit0", spark_bitwise_and(spark_shift_right(F.col("idx"), q0), F.lit(1)))
            .withColumn("bit1", spark_bitwise_and(spark_shift_right(F.col("idx"), q1), F.lit(1)))
            .withColumn("qubit_bits", 
                spark_bitwise_or(F.col("bit0"), spark_shift_left(F.col("bit1"), 1)).cast("int"))
            .join(gate_df, F.col("qubit_bits") == F.col("g_col"), "inner")
            .withColumn("cleared_idx",
                spark_bitwise_xor(
                    spark_bitwise_xor(F.col("idx"), spark_shift_left(F.col("bit0"), q0)),
                    spark_shift_left(F.col("bit1"), q1)
                ))
            .withColumn("new_bit0", spark_bitwise_and(F.col("g_row"), F.lit(1)))
            .withColumn("new_bit1", spark_bitwise_and(spark_shift_right(F.col("g_row"), 1), F.lit(1)))
            .withColumn("new_idx",
                spark_bitwise_or(
                    spark_bitwise_or(F.col("cleared_idx"), spark_shift_left(F.col("new_bit0"), q0)),
                    spark_shift_left(F.col("new_bit1"), q1)
                ))
            .withColumn("new_real",
                F.col("g_real") * F.col("real") - F.col("g_imag") * F.col("imag"))
            .withColumn("new_imag",
                F.col("g_real") * F.col("imag") + F.col("g_imag") * F.col("real"))
            .groupBy("new_idx")
            .agg(F.sum("new_real").alias("real"), F.sum("new_imag").alias("imag"))
            .select(F.col("new_idx").alias("idx"), "real", "imag")
            .filter((F.abs(F.col("real")) > 1e-15) | (F.abs(F.col("imag")) > 1e-15))
        )
        
        if self.num_partitions > 1:
            result = result.repartition(self.num_partitions, "idx")
        
        return result
    
    def apply_level_parallel(
        self, 
        state_df: DataFrame, 
        gates: List[Gate], 
        level_indices: List[int]
    ) -> DataFrame:
        """
        Apply all gates in a level with TRUE parallel execution.
        
        Gates on independent qubits are fused into a single transformation.
        Gates on overlapping qubits are applied sequentially within groups.
        """
        level_gates = [gates[i] for i in level_indices]
        
        if not level_gates:
            return state_df
        
        # Group gates by qubit independence
        groups = self._group_independent_gates(level_gates)
        
        current_state = state_df
        for group in groups:
            if len(group) == 1:
                current_state = self._apply_single_gate(current_state, group[0])
            else:
                # Apply independent gates in parallel!
                current_state = self.apply_gates_parallel(current_state, group)
        
        return current_state
    
    def _group_independent_gates(self, gates: List[Gate]) -> List[List[Gate]]:
        """
        Group gates into sets of independent gates.
        
        Gates in the same group operate on non-overlapping qubits.
        """
        groups = []
        current_group = []
        current_qubits: Set[int] = set()
        
        for gate in gates:
            gate_qubits = set(gate.qubits)
            
            if gate_qubits & current_qubits:
                # Overlap - start new group
                if current_group:
                    groups.append(current_group)
                current_group = [gate]
                current_qubits = gate_qubits
            else:
                # No overlap - add to current group
                current_group.append(gate)
                current_qubits.update(gate_qubits)
        
        if current_group:
            groups.append(current_group)
        
        return groups
