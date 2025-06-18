


# bash jobs/SEEOOD_baselines/summarize.sh > jobs/SEEOOD_baselines/summary.txt
python jobs/SEEOOD_baselines/summarize.py --experiment mnist
python jobs/SEEOOD_baselines/summarize.py --experiment fashionmnist
python jobs/SEEOOD_baselines/summarize.py --experiment mnist-fashionmnist
python jobs/SEEOOD_baselines/summarize.py --experiment svhn
python jobs/SEEOOD_baselines/summarize.py --experiment cifar10-svhn

