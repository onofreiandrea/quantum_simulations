# AWS Spark Cluster Setup for wenbo_engine

## Architecture

```
                    ┌──────────────────┐
                    │   Your laptop    │
                    │  (SSH to master) │
                    └────────┬─────────┘
                             │ SSH
                    ┌────────▼─────────┐
                    │   EC2 Master     │
                    │  c6i.4xlarge     │
                    │  Spark driver    │
                    └──┬────┬────┬─────┘
              ┌────────┘    │    └────────┐
              ▼             ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker 1 │ │ Worker 2 │ │ Worker 3 │
        │c6i.4xlrg │ │c6i.4xlrg │ │c6i.4xlrg │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
             └──────┬──────┘─────────────┘
                    │  NFS mount
              ┌─────▼──────┐
              │  Amazon EFS │
              │  /mnt/efs   │
              │  (chunks)   │
              └─────────────┘
```

## Step 1: Create Security Group

In the AWS Console → EC2 → Security Groups → Create:

- **Name:** `wenbo-spark-cluster`
- **Inbound rules:**
  | Type       | Port      | Source               | Purpose           |
  |------------|-----------|----------------------|-------------------|
  | SSH        | 22        | Your IP              | SSH access        |
  | Custom TCP | 7077      | Security group self  | Spark master      |
  | Custom TCP | 8080      | Your IP              | Spark Web UI      |
  | Custom TCP | 0-65535   | Security group self  | Spark internal    |
  | NFS        | 2049      | Security group self  | EFS mount         |

**"Security group self"** = set Source to the security group's own ID so all nodes in the cluster can talk to each other.

## Step 2: Create EFS Filesystem

AWS Console → EFS → Create file system:

1. **Name:** `wenbo-state-vectors`
2. **VPC:** Same VPC as your EC2 instances
3. **Performance:** General Purpose (or Max I/O for 38+ qubits)
4. **Throughput:** Bursting (or Provisioned if you know the workload)
5. Click **Create**
6. Go to **Network** tab → ensure mount targets exist in your subnet
7. **Security group** on mount targets: use `wenbo-spark-cluster`
8. Note the **DNS name**: `fs-XXXXXXXX.efs.<region>.amazonaws.com`

### Storage costs

EFS charges ~$0.30/GB/month. A 35-qubit run uses ~512 GB = ~$150/month if left on disk. **Delete work directories after extracting results.**

## Step 3: Launch EC2 Instances

Launch **4 instances** (1 master + 3 workers):

1. AWS Console → EC2 → Launch Instance
2. **AMI:** Ubuntu Server 22.04 LTS (x86_64)
3. **Instance type:** `c6i.4xlarge` (16 vCPU, 32 GB RAM, ~$0.68/hr)
4. **Key pair:** Create or select your SSH key
5. **Network:** Same VPC/subnet as EFS
6. **Security group:** `wenbo-spark-cluster`
7. **Storage:** 30 GB gp3 (just for OS + Spark + Python — state data goes on EFS)
8. **Number of instances:** 4
9. **Name tags:** `wenbo-master`, `wenbo-worker-1`, `wenbo-worker-2`, `wenbo-worker-3`

### Cost estimate (eu-central-1)

| Resource         | Cost/hr    | 4 nodes    |
|------------------|-----------|------------|
| c6i.4xlarge × 4  | $0.68/ea  | $2.72/hr   |
| EFS (512 GB)     |           | ~$0.02/hr  |
| **Total**        |           | **~$2.74/hr** |

**Remember to stop instances when not in use!**

## Step 4: Setup Each Node

SSH into each instance and run:

```bash
# Copy deploy files to the master first
scp -i your-key.pem -r wenbo_engine/deploy/ ubuntu@<master-ip>:~/deploy/

# Then on EACH node (master + all workers):
ssh -i your-key.pem ubuntu@<node-ip>

# Run setup (replace with your EFS DNS name)
chmod +x ~/deploy/ec2_setup.sh
~/deploy/ec2_setup.sh fs-XXXXXXXX.efs.eu-central-1.amazonaws.com
```

## Step 5: Upload Project Code

From your laptop:

```bash
# Copy the entire project to EFS (via master)
scp -i your-key.pem -r wenbo_engine/ ubuntu@<master-ip>:/mnt/efs/quantum_simulations/wenbo_engine/
```

Since EFS is mounted on all nodes, the code is immediately available everywhere.

## Step 6: Start Spark Cluster

The master runs Spark master + driver + NFS only — **no executor**.
Only the 2 worker nodes run executors. This prevents OOM on the master.

```bash
# On MASTER (Spark master + NFS only, NO executor):
ssh -i your-key.pem ubuntu@<master-ip>
chmod +x /mnt/efs/quantum_simulations/wenbo_engine/deploy/spark_cluster.sh
/mnt/efs/quantum_simulations/wenbo_engine/deploy/spark_cluster.sh start-master
# Note the master IP printed

# On EACH WORKER (NOT on master):
ssh -i your-key.pem ubuntu@<worker-ip>
/mnt/efs/quantum_simulations/wenbo_engine/deploy/spark_cluster.sh start-worker <master-private-ip>
```

Verify at `http://<master-public-ip>:8080` — you should see 2 workers registered.

## Step 7: Run Simulation

On the **master** node:

```bash
cd /mnt/efs/quantum_simulations

# Dry run first (prints config, no simulation)
spark-submit --master spark://<master-private-ip>:7077 \
    wenbo_engine/deploy/run_distributed.py \
    --qubits 35 --circuit ghz --dry-run

# Actual run
spark-submit --master spark://<master-private-ip>:7077 \
    --py-files wenbo_engine/ \
    wenbo_engine/deploy/run_distributed.py \
    --qubits 35 --circuit ghz --chunk-size 20
```

### Recommended chunk sizes

| Qubits | Chunk size (log2) | Chunk bytes | Num chunks | Notes                |
|--------|-------------------|-------------|------------|----------------------|
| 30     | 18                | 2 MB        | 4,096      | Quick test           |
| 35     | 20                | 8 MB        | 32,768     | Good default         |
| 37     | 20                | 8 MB        | 131,072    | ~1 TB state          |
| 40     | 22                | 32 MB       | 262,144    | Needs Lustre/big EFS |

### Start small!

Test with 25-30 qubits first to validate the setup, then scale up:

```bash
# Quick validation (seconds, ~8 MB state)
spark-submit --master spark://<master-private-ip>:7077 \
    wenbo_engine/deploy/run_distributed.py \
    --qubits 20 --circuit ghz --chunk-size 16

# Medium test (minutes, ~8 GB state)
spark-submit --master spark://<master-private-ip>:7077 \
    wenbo_engine/deploy/run_distributed.py \
    --qubits 30 --circuit ghz --chunk-size 18

# Full 35-qubit run
spark-submit --master spark://<master-private-ip>:7077 \
    wenbo_engine/deploy/run_distributed.py \
    --qubits 35 --circuit ghz --chunk-size 20
```

## Cleanup

**IMPORTANT:** Stop/terminate instances and delete EFS data when done to avoid charges.

```bash
# Delete simulation data from EFS
rm -rf /mnt/efs/wenbo_work/run_*

# Stop instances (AWS Console or CLI)
aws ec2 stop-instances --instance-ids i-xxx i-yyy i-zzz i-www

# Or terminate if you're done for good
aws ec2 terminate-instances --instance-ids i-xxx i-yyy i-zzz i-www
```

## Troubleshooting

### EFS mount fails
- Check security group allows NFS (port 2049) between instances
- Check EFS mount target exists in your subnet

### Workers don't connect to master
- Check security group allows port 7077 between instances
- Use **private IPs** for intra-cluster communication
- Check `$SPARK_HOME/logs/` for errors

### Out of disk space
- EFS is elastic (unlimited), but check gp3 root volume isn't full
- `df -h` to check

### Simulation is slow
- Check Spark UI at `:8080` for task distribution
- Increase chunk_size if I/O bound (many small files = overhead)
- Check EFS throughput mode — switch to Provisioned if hitting burst limits

### Recovery after crash
- The WAL handles this automatically — just re-run the same command
- It will resume from the last committed step
