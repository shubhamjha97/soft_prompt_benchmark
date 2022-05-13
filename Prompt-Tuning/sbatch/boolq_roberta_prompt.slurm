#!/bin/bash
#SBATCH --job-name=PromptTuning-boolq-roberta
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ask9126/%j_%x.out
#SBATCH --error=/scratch/ask9126/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=18:00:00
#SBATCH --gres=gpu
#SBATCH --mem=28G
#SBATCH -c 4
#SBATCH --requeue

singularity exec --nv --bind /scratch --overlay /scratch/ask9126/overlay-25GB-500K.ext3:ro /scratch/ask9126/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda create -n pt2 python=3.8.5
conda activate pt2
cd /scratch/ask9126/soft_prompt_benchmark/Prompt-Tuning
pip install -r requirements.txt
python3 -u train.py 'task_name=boolq' 'plm=roberta-large' 'method=prompt_tuning'
"
