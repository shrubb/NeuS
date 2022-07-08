# Usage:
# bash run-tensorboard.sh <logdir> <port>
LOGDIR=$1
PORT=$2

PYTHONPATH= PYTHONHOME= nohup srun -c 2 -p cpu,gpu,mem --time 6-0 --job-name tensorboard-${PORT} conda run --no-capture-output -n Tensorboard-nightly tensorboard --logdir ${LOGDIR} --port ${PORT} --samples_per_plugin scalars=1000,images=100 --bind_all --load_fast true > log-tensorboard-${PORT}.txt 2>&1 &

