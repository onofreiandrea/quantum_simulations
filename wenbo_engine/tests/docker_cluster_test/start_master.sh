#!/bin/bash
set -e

/opt/spark/sbin/start-master.sh --host spark-master --port 7077

cd /code
python3 -c "
import zipfile, os
with zipfile.ZipFile('wenbo_engine.zip','w') as z:
    for r,ds,fs in os.walk('wenbo_engine'):
        ds[:] = [d for d in ds if d != '__pycache__']
        for f in fs:
            if not f.endswith('.pyc'):
                z.write(os.path.join(r,f))
print('zip created')
"

echo "Waiting for workers to register..."
sleep 10

spark-submit \
    --master spark://spark-master:7077 \
    --py-files /code/wenbo_engine.zip \
    /code/wenbo_engine/tests/docker_cluster_test/run_test.py
