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
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/vos/classification/CIFAR/jobs/im32-eval.log

# python train_customized.py --start_epoch 40 --sample_number 1000 --sample_from 10000 --select 1 --loss_weight 0.1 --dataset imagenet10 --nf 32 --prefetch 16 --learning_rate 0.1

# Eval
python test_customized.py --dataset imagenet10 --score energy --method_name imagenet10_dense_baseline_dense_0.1_1000_40_1_10000  --num_to_avg 10 --model_name dense --nf 32 --prefetch 16
python test_customized.py --dataset imagenet10 --score energy --method_name imagenet10-32-o-01  --num_to_avg 10 --model_name dense --nf 32 --prefetch 16 --test_energy_baseline
