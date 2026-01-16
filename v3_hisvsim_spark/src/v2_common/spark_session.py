"""
SparkSession factory for the quantum simulator.
"""
from __future__ import annotations

from pyspark.sql import SparkSession

from v2_common.config import SimulatorConfig, DEFAULT_CONFIG
# Also allow direct import
try:
    from config import SimulatorConfig as Config, DEFAULT_CONFIG as DefaultConfig
except ImportError:
    pass


def create_spark_session(config: SimulatorConfig = DEFAULT_CONFIG) -> SparkSession:
    """
    Create and configure a SparkSession for quantum simulation.
    
    Args:
        config: Simulator configuration with Spark settings.
        
    Returns:
        Configured SparkSession instance.
    """
    builder = (
        SparkSession.builder
        .appName(config.spark_app_name)
        .master(config.spark_master)
        .config("spark.sql.shuffle.partitions", config.spark_shuffle_partitions)
        .config("spark.driver.memory", config.spark_driver_memory)
        .config("spark.executor.memory", config.spark_executor_memory)
        # Optimize for our workload
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Parquet settings
        .config("spark.sql.parquet.compression.codec", "snappy")
        # Broadcast settings (gate matrices are small)
        .config("spark.sql.autoBroadcastJoinThreshold", "100MB")
    )
    
    return builder.getOrCreate()


def get_or_create_spark_session(config: SimulatorConfig = DEFAULT_CONFIG) -> SparkSession:
    """
    Get existing SparkSession or create a new one.
    
    Args:
        config: Simulator configuration with Spark settings.
        
    Returns:
        SparkSession instance.
    """
    existing = SparkSession.getActiveSession()
    if existing is not None:
        return existing
    return create_spark_session(config)
