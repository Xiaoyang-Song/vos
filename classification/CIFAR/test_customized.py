import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
# from models.wrn import WideResNet
# from models.densenet import DenseNet3
from models.wrn_virtual import WideResNet
from models.densenet import DenseNet3
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from models.densenet_GP import DenseNet3GP
from dataset import *
from sklearn.metrics import roc_auc_score

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    import utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--dataset', type=str, default='CIFAR10-SVHN')
parser.add_argument('--num_to_avg', type=int, default=1,
                    help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true',
                    help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true',
                    help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str,
                    default='cifar10_allconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers')
parser.add_argument('--nf', type=int, default=32)
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float,
                    help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=1,
                    help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true',
                    help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str,
                    help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float,
                    help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--model_name', default='res', type=str)
# Customized
parser.add_argument('--test_energy_baseline', action='store_true', help='Test energy baseline or not')

args = parser.parse_args()
print(args)
# torch.manual_seed(1)
# np.random.seed(1)

if args.dataset == 'cifar10':
    # mean and standard deviation of channels of CIFAR-10 images
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                   trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    train_data = dset.CIFAR10('./Dataset/CIFAR-10',
                              train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10('./Dataset/CIFAR-10',
                             train=False, transform=test_transform, download=True)
    num_classes = 10
    num_channels = 3
    num_features = 64

elif args.dataset == 'mnist':
    transform = transforms.Compose([ transforms.Resize((32, 32)), 
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST("./Datasets", download=True, transform=transform)
    test_data = torchvision.datasets.MNIST("./Datasets", download=True, train=False, transform=transform)
    num_classes = 10
    num_channels = 3
    num_features = 32

elif args.dataset == 'imagenet10':
    train_set, test_set = imagenet10_set_loader(256, 0)
    total_size = len(train_set)
    train_ratio = 0.8
    val_ratio = 0.2
    print('Total dataset size: ', total_size)
    # Calculate sizes for each split
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    if train_size + val_size != total_size:
        val_size = val_size + 1 # This is specifically for imagenet100

    # Perform the split
    train_data, validation_data = torch.utils.data.random_split(train_set, [train_size, val_size])
    print("Dataset size: ", len(train_data), len(validation_data), len(test_set))
    test_data = validation_data + test_set
    num_classes = 10
    num_channels = 3
    num_features = args.nf

elif args.dataset == 'SVHN' or args.dataset == 'FashionMNIST':
    data = DSET(args.dataset, True, 128, 128, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    train_data, test_data = data.ind_train, data.ind_val
    num_classes = 8

    if args.dataset == 'SVHN':
        num_channels = 3
    else:
        num_channels = 1

elif args.dataset == 'MNIST':
    data = DSET(args.dataset, True, 128, 128, [2, 3, 6, 8, 9], [1, 7])
    train_data, test_data = data.ind_train, data.ind_val
    num_channels = 1
    num_classes = 5
else:
    assert False


TPR = 0.95
# Imagenet10
if args.dataset == 'imagenet10':
    val_size = 1500
    ood_test_size = 1600
    ind_test_size = 1600
elif args.dataset == 'mnist':
    # MNIST
    val_size = 2000
    ood_test_size = 2000
    ind_test_size = 2000

val_data = torch.utils.data.Subset(test_data, range(val_size))
test_data = torch.utils.data.Subset(test_data, range(val_size, len(test_data)))
print(len(val_data), len(test_data))
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)


# Create model
if args.model_name == 'res':
    net = WideResNet(args.layers, num_classes,
                     args.widen_factor, dropRate=args.droprate)
else:
    net = DenseNet3GP(100, num_classes, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0, num_channels= num_channels, feature_size=num_features)
    print("Model checkpoint loaded successfully.")
start_epoch = 0

# ENERGY
if args.test_energy_baseline:
    experiment = args.method_name
    pre_trained_net = f"/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/ckpt/{experiment}/densenet_{args.dataset}.pth"
    net.load_state_dict(torch.load(pre_trained_net))

#VOS
else:
    # Restore model
    if args.load != '':
        for i in range(1000 - 1, -1, -1):
            if 'pretrained' in args.method_name:
                subdir = 'pretrained'
            elif 'oe_tune' in args.method_name:
                subdir = 'oe_tune'
            elif 'energy_ft' in args.method_name:
                subdir = 'energy_ft'
            elif 'baseline' in args.method_name:
                subdir = 'baseline'
            else:
                subdir = 'oe_scratch'

            model_name = os.path.join(os.path.join(
                args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt')
            # model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '.pt')
            if os.path.isfile(model_name):
                net.load_state_dict(torch.load(model_name))
                print('Model restored! Epoch:', i)
                start_epoch = i + 1
                break
        if start_epoch == 0:
            assert False, "could not resume " + model_name

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))


def concat(x): return np.concatenate(x, axis=0)
def to_np(x): return x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            #     break

            data = data.cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(
                    to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':
                    _score.append(-to_np((args.T *
                                  torch.logsumexp(output / args.T, dim=1))))
                # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                else:
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(
                        to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(
                        to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()



# ENERGY: VOS uses this score function
val_in_scores, _, _ = get_ood_scores(val_loader, in_dist=True)
print("InD Val size: ", len(val_in_scores))
threshold = np.quantile(val_in_scores, 1 - TPR)  # Threshold at 5% FPR
# Test on InD
test_in_scores, test_right_scores, test_wrong_scores = get_ood_scores(test_loader, in_dist=True)
test_in_scores = test_in_scores[0:ind_test_size]
print("InD Testing size: ", len(test_in_scores))
test_in_correct = test_in_scores >= threshold
test_in_accuracy = np.mean(test_in_correct)
print(f"Test In-Distribution Accuracy: {test_in_accuracy * 100:.2f}%")

# TESTING OOD data
if args.dataset == 'mnist':
    print('######################################')
    print('Testing on FashionMNIST') 
    transform = transforms.Compose([ transforms.Resize((32, 32)), 
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor()])
    tset = torchvision.datasets.FashionMNIST("./Datasets", download=True, train=True, transform=transform)
    ood_loader = torch.utils.data.DataLoader(tset, batch_size=args.test_bs, shuffle=False, num_workers=1, pin_memory=True)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")

    print('######################################')
    print('Testing on CIFAR10')
    normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(), normalizer])
    tset = datasets.CIFAR10('./Datasets/CIFAR-10', train=True, download=True, transform=transform)
    ood_loader = torch.utils.data.DataLoader(tset, batch_size=args.test_bs, shuffle=False, num_workers=1, pin_memory=True)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")

    print('######################################')
    print('Testing on SVHN')
    transform = transforms.Compose([transforms.ToTensor()])
    tset = datasets.SVHN('./Datasets/SVHN', split='test', download=True, transform=transform)
    ood_loader = torch.utils.data.DataLoader(tset, batch_size=args.test_bs, shuffle=False, num_workers=1)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")

    print('######################################')
    print('Testing on ImageNet-c') 
    transform = transforms.Compose([transforms.RandomCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    tset = datasets.ImageFolder(os.path.join('../../../GP-ImageNet/data/Imagenet'), transform=transform)
    ood_loader = torch.utils.data.DataLoader(tset, batch_size=args.test_bs, shuffle=False, num_workers=1)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")

elif args.dataset == 'imagenet10':
    print('######################################')
    print('Testing on LSUN-C')
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    data = torchvision.datasets.ImageFolder(root="../../../GP-ImageNet/data/LSUN/",
                                transform=transforms.Compose([transforms.Resize((32, 32)), 
                                                            transforms.CenterCrop(32), 
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(data, batch_size=args.test_bs, shuffle=False, num_workers=1)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")


    print('######################################')
    print('Testing on LSUN-R')
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    data = torchvision.datasets.ImageFolder(root="../../../GP-ImageNet/data/LSUN_resize/",
                                transform=transforms.Compose([transforms.Resize((32, 32)), 
                                                            transforms.CenterCrop(32), 
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(data, batch_size=args.test_bs, shuffle=False, num_workers=1)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")

    print('######################################')
    print('Testing on iSUN')
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    data = torchvision.datasets.ImageFolder(root="../../../GP-ImageNet/data/iSUN/",
                                transform=transforms.Compose([transforms.Resize((32, 32)), 
                                                            transforms.CenterCrop(32), 
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(data, batch_size=args.test_bs, shuffle=False, num_workers=1)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")

    print('######################################')
    print('Testing on Places365')
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    data = datasets.Places365(root="../../../GP-ImageNet/data/", split='val', small=True, download=False, 
                            transform=transforms.Compose([transforms.Resize((32, 32)), 
                                                            transforms.CenterCrop(32), 
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean, std)]))
    data = torch.utils.data.Subset(data, range(5000))
    ood_loader = torch.utils.data.DataLoader(data, batch_size=args.test_bs, shuffle=False, num_workers=16)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")

    print('######################################')
    print('Testing on DTD Texture')
    data = torchvision.datasets.ImageFolder(root="../../../GP-ImageNet/data/dtd/images/",
                                transform=transforms.Compose([transforms.Resize((32, 32)), transforms.CenterCrop(32), 
                                                                transforms.ToTensor(), 
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))
    ood_loader = torch.utils.data.DataLoader(data, batch_size=args.test_bs, shuffle=False, num_workers=16)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")

    print('######################################')
    print('Testing on SVHN')
    transform = transforms.Compose([transforms.ToTensor()])
    tset = datasets.SVHN('./Datasets/SVHN', split='test', download=True, transform=transform)
    ood_loader = torch.utils.data.DataLoader(tset, batch_size=args.test_bs, shuffle=False, num_workers=1)
    ood_scores = get_ood_scores(ood_loader, in_dist=False)
    ood_scores = ood_scores[val_size:val_size + ood_test_size]
    print("OOD Test size: ", len(ood_scores))
    ood_correct = ood_scores < threshold
    ood_accuracy = np.mean(ood_correct)
    print(f"OOD Detection Accuracy: {ood_accuracy * 100:.2f}%")
    all_scores = np.concatenate([test_in_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_in_scores)), np.zeros(len(ood_scores))])  # 1 for in-dist, 0 for OOD
    auroc = roc_auc_score(all_labels, all_scores)  # Use negative scores if lower scores indicate OOD
    print(f"AUROC: {auroc * 100:.2f}%")
