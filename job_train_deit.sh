#!/bin/bash 
#SBATCH --job-name=ThinViT
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256000
#SBATCH --time=7-00:00:00
#SBATCH --output=alideit.out
#SBATCH --error=alideit.err
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:4


module load gpu-env
module load cuda
module load gcc12-env
module load python

source /home/aho/envs/env/bin/activate



./distributed_train.sh 4 --config args.yaml --model thinkingvit --eval-every 10 --batch-size 256 --data '/data22/datasets/ilsvrc2012/' --initial-checkpoint /home/aho/thinkingvit/tiny_in_base.pth.tar --thinking_stages 3 6 &> ThinkingViT.log




jobinfo
