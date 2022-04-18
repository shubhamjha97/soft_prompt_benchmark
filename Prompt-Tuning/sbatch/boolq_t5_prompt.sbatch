#!/bin/bash
#SBATCH --job-name=t5_boolq_prompt
#SBATCH --open-mode=append
#SBATCH --output=/scratch/sj3549/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/sj3549/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4

singularity exec --nv --bind /scratch --overlay /scratch/sj3549/nlu-overlay.ext3:ro /scratch/sj3549/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/miniconda3/bin/activate
conda activate project_env
cd /scratch/sj3549/soft_prompt_benchmark/Prompt-Tuning
python3 -u train.py 'task_name=boolq' 'plm=t5-base' 'method=prompt_tuning'
"
