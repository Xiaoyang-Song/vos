#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=fmvos
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/vos/classification/CIFAR/jobs/out/vos-fashionmnist.log

python train_virtual_dense.py --start_epoch 40 --sample_number 1000 \
    --sample_from 10000 --select 1 --loss_weight 0.1 --dataset FashionMNIST \
    --prefetch 16 --learning_rate 0.1

python test.py --model_name dense --method_name FashionMNIST_dense_baseline_dense_0.1_1000_40_1_10000 \
    --score energy --num_to_avg 5 --dataset FashionMNIST