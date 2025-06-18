# sbatch jobs/imagenet32.sh
# sbatch jobs/mnist32.sh


# SEE-OOD paper baselines
sbatch jobs/SEEOOD_baselines/mnist.sh
sbatch jobs/SEEOOD_baselines/fashionmnist.sh
sbatch jobs/SEEOOD_baselines/mnist-fashionmnist.sh
sbatch jobs/SEEOOD_baselines/svhn.sh
sbatch jobs/SEEOOD_baselines/cifar10-svhn.sh