#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=90GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=gpp-llie
#SBATCH --output=log_gpp-llie/gpp-llie_%j.txt
#SBATCH --error=log_gpp-llie/gpp-llie_%j.err
#SBATCH --account=pr_107_general

# --- Load modules ---
module purge

# --- Paths ---
CONDA_ENV_PATH="/scratch/ll5484/conda/envs/gpp-llie"
SCRIPT_PATH="train_dit.py"

# --- Create log directory if missing ---
mkdir -p log_gpp-llie

# --- Run function ---
run_experiment() {
    singularity exec --nv \
        --overlay /scratch/ll5484/dinov2/overlay-50G-10M.ext3:ro \
        /scratch/ll5484/dinov2/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
        /bin/bash -c "
            source /ext3/env.sh
            conda activate $CONDA_ENV_PATH
            python -u $SCRIPT_PATH
        "
}

# --- Execute the experiment ---
run_experiment
