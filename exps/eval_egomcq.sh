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

JOB_NAME='eval_egomcq'
PARTITION='HOD'

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:1 \
    -u python evaluation/eval_egomcq.py \
    --config_file configs/no_decoder/clip_base_eval.yml \
    --root ego4d/videos_short320_chunked_15s/ \
    --metadata annotations/egomcq.json \
    --crop_size 224