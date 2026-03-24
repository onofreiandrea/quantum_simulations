FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip wget default-jre-headless procps && \
    pip3 install numpy pyspark==3.5.8 pytest && \
    wget -q "https://dlcdn.apache.org/spark/spark-3.5.8/spark-3.5.8-bin-hadoop3.tgz" -O /tmp/spark.tgz && \
    tar xzf /tmp/spark.tgz -C /opt && \
    ln -s /opt/spark-3.5.8-bin-hadoop3 /opt/spark && \
    rm /tmp/spark.tgz && \
    apt-get clean

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
ENV PYSPARK_PYTHON=python3
