#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=IM32VOS
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/vos/classification/CIFAR/jobs/im32.log

python train_customized.py --start_epoch 40 --sample_number 1000 --sample_from 10000 --select 1 --loss_weight 0.1 --dataset imagenet10 --nf 32 --prefetch 16 --learning_rate 0.1