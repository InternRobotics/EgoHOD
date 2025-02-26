export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_PORT=$((10000 + $RANDOM % 100))

JOB_NAME='pretrain'
PARTITION='HOD'
GPUS=16
GPUS_PER_NODE=8
CPUS_PER_TASK=12
NNODE=1
JOB_DIR='./log/'

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:8 \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --cpus-per-task=${CPUS_PER_TASK} \
    -u python main_pretrain.py \
    --config_file configs/no_decoder/debug_clip_large.yml \