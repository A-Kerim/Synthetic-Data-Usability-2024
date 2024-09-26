import os
import pickle
from collections import defaultdict

import torch
import numpy as np
import random
import torch.utils.data
from torch.utils.data import random_split, DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
# Set the random seed for reproducibility
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Synth train+valid =  844
# Real train+val = 844
# Real test = 900

# This approach is just to overfit to the synthetic training data
def train_synthetic_ourApproach(ds_name, batch_size,trans_train,trans_val_test,is_hec):
    # should overvit to synthetic data
    print("train_synthetic_ourApproach")
    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets/car_accidents', transform=trans_train)
    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets/car_accidents', transform=trans_train)


    n_train = int(len(cars_synth) * 0.90)

    n_valid = len(cars_synth) - n_train
    cars_synth_train, cars_synth_val = random_split(cars_synth, [n_train, n_valid])


    train_loader = DataLoader(cars_synth_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(cars_synth_val, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(cars_synth_val, batch_size=batch_size, num_workers=8)

    return train_loader, val_loader, test_loader

def train_synthetic_only(ds_name,batch_size,trans_train,trans_val_test,is_hec,ds_perc):
    print("train_synthetic_only-dsname:{}".format(ds_name))
    # ["car_accidents", "cifar", "birds"]
    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)


    n_train = int(len(cars_synth) * 0.90)

    # this will help us accomodate variable dataset training
    n_train = int(n_train*ds_perc)

    n_valid = len(cars_synth) - n_train
    n_valid = int(n_valid * ds_perc)
    cars_synth_train, cars_synth_val,_ = random_split(cars_synth, [n_train, n_valid, len(cars_synth)-(n_train+n_valid)])

    _, cars_real_val, _ = random_split(cars_real_tr_val, [n_train, n_valid, len(cars_real_tr_val)-(n_train+n_valid)])

    train_loader = DataLoader(cars_synth_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(cars_real_val, batch_size=batch_size,shuffle=True, num_workers=8)
    test_loader = DataLoader(cars_real_test, batch_size=batch_size,shuffle=True, num_workers=8)

    return [train_loader], val_loader, test_loader


def train_synthetic_only_nonphoto(ds_name, batch_size,trans_train,trans_val_test,is_hec,ds_perc,):
    print("train_synthetic_only_nonphoto")
    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets_nonphoto/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets_nonphoto/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)


    # Assuming cars_synth and cars_real_tr_val are PyTorch datasets and ds_perc is a float

    # Calculate the number of training samples from cars_synth based on ds_perc
    num_train_samples = int(len(cars_synth) * ds_perc)

    # Calculate the number of validation samples (30% of num_train_samples)
    num_val_samples = int(num_train_samples * 0.30)

    # Ensure you don't exceed the length of cars_real_tr_val
    num_val_samples = min(num_val_samples, len(cars_real_tr_val))

    # Split the cars_real_tr_val dataset
    cars_real_val, _ = random_split(cars_real_tr_val, [num_val_samples, len(cars_real_tr_val) - num_val_samples])

    # Split the cars_synth dataset to get the training set
    cars_synth_train, _ = random_split(cars_synth, [num_train_samples, len(cars_synth) - num_train_samples])

    train_loader = DataLoader(cars_synth_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(cars_real_val, batch_size=batch_size,shuffle=True, num_workers=8)
    test_loader = DataLoader(cars_real_test, batch_size=batch_size,shuffle=True, num_workers=8)

    return [train_loader], val_loader, test_loader

def train_real_only(ds_name, batch_size,trans_train,trans_val_test, is_hec,ds_perc):
    print("train_real_only")

    if is_hec:
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
    else:
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)


    n_train = int(len(cars_real_tr_val) * 0.90)

    # this will help us accomodate variable dataset training
    n_train = int(n_train*ds_perc)

    n_valid = len(cars_real_tr_val) - n_train
    n_valid = int(n_valid * ds_perc)

    cars_real_train, cars_real_val, _ = random_split(cars_real_tr_val, [n_train,n_valid, len(cars_real_tr_val)-(n_train + n_valid)])

    train_loader = DataLoader(cars_real_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(cars_real_train,  batch_size=batch_size,shuffle=True, num_workers=8)
    test_loader = DataLoader(cars_real_train, batch_size=batch_size,shuffle=True, num_workers=8)

    return [train_loader], val_loader, test_loader


def train_synthetic_real(ds_name, batch_size,trans_train,trans_val_test,is_hec):
    print("train_synthetic_real [0.5 0.5]")
    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)


    # because we will combine samples from synth and real
    n_train = int(len(cars_real_tr_val) * 0.90)


    cars_real_train, cars_real_val = random_split(cars_real_tr_val, [int(n_train*0.5), len(cars_real_tr_val)-int(n_train*0.5)])

    cars_synth_train, cars_synth_val = random_split(cars_synth, [int(n_train*0.5), len(cars_synth)-int(n_train*0.5)])

    train_loader = DataLoader(ConcatDataset([cars_synth_train, cars_real_train]),  batch_size=batch_size, shuffle=True,
                              num_workers=8)
    val_loader = DataLoader(cars_real_val,   batch_size=batch_size,shuffle=True,
                            num_workers=8)
    test_loader = DataLoader(cars_real_test, batch_size=batch_size,shuffle=True,
                             num_workers=8)

    return train_loader, val_loader, test_loader


def train_synthetic_real_nonphoto(ds_name, batch_size,trans_train,trans_val_test,is_hec):
    print("train_synthetic_real_nonphoto [0.5 0.5]")
    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets_nonphoto/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets_nonphoto/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)

    # because we will combine samples from synth and real
    n_train = int(len(cars_real_tr_val) * 0.90)


    cars_real_train, cars_real_val = random_split(cars_real_tr_val, [int(n_train*0.5), len(cars_real_tr_val)-int(n_train*0.5)])

    cars_synth_train, cars_synth_val = random_split(cars_synth, [int(n_train*0.5), len(cars_synth)-int(n_train*0.5)])

    train_loader = DataLoader(ConcatDataset([cars_synth_train, cars_real_train]),  batch_size=batch_size, shuffle=True,
                              num_workers=8)
    val_loader = DataLoader(cars_real_val,   batch_size=batch_size,shuffle=True,
                            num_workers=8)
    test_loader = DataLoader(cars_real_test, batch_size=batch_size,shuffle=True,
                             num_workers=8)

    return train_loader, val_loader, test_loader

def train_syntheticPLS_real(ds_name, batch_size,trans_train,trans_val_test,is_hec):
    print("train_syntheticPLS_real [0.75 0.25]")

    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)

    # because we will combine samples from synth and real
    n_train = int(len(cars_real_tr_val) * 0.90)


    cars_real_train, cars_real_val = random_split(cars_real_tr_val, [int(n_train*0.25), len(cars_real_tr_val)-int(n_train*0.25)])

    cars_synth_train, cars_synth_val = random_split(cars_synth, [int(n_train*0.75), len(cars_synth)-int(n_train*0.75)])

    train_loader = DataLoader(ConcatDataset([cars_synth_train, cars_real_train]),  batch_size=batch_size, shuffle=True,
                              num_workers=8)
    val_loader = DataLoader(cars_real_val,   batch_size=batch_size,shuffle=True,
                            num_workers=8)
    test_loader = DataLoader(cars_real_test, batch_size=batch_size,shuffle=True,
                             num_workers=8)

    return train_loader, val_loader, test_loader

def train_synthetic_realPLS(ds_name, batch_size,trans_train,trans_val_test,is_hec):
    print("train_synthetic_realPLS [0.25 0.75]")

    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)

    # because we will combine samples from synth and real
    n_train = int(len(cars_real_tr_val) * 0.90)


    cars_real_train, cars_real_val = random_split(cars_real_tr_val, [int(n_train*0.75), len(cars_real_tr_val)-int(n_train*0.75)])

    cars_synth_train, cars_synth_val = random_split(cars_synth, [int(n_train*0.25), len(cars_synth)-int(n_train*0.25)])

    train_loader = DataLoader(ConcatDataset([cars_synth_train, cars_real_train]),  batch_size=batch_size, shuffle=True,
                              num_workers=8)
    val_loader = DataLoader(cars_real_val,   batch_size=batch_size,shuffle=True,
                            num_workers=8)
    test_loader = DataLoader(cars_real_test, batch_size=batch_size,shuffle=True,
                             num_workers=8)

    return train_loader, val_loader, test_loader



# def get_top_n_paths(basePath_hec, basePath_pc,scores_temp, metric_type, is_hec,n=-1):
#     # List to hold all paths and their corresponding scores for the given metric type
#     all_paths_metrics = [(path, metrics[metric_type]) for path, metrics in scores_temp.items()]
#     for index in range(len(all_paths_metrics)):
#         if is_hec:  # for HEC
#             all_paths_metrics[index] = basePath_hec + all_paths_metrics[index][0].replace(basePath_pc, "")
#         else:
#             # Always the scores should be corrected to reflect Lab Workstation
#             pass
#
#         # Sort all images by their metric score in descending order
#     sorted_images = sorted(all_paths_metrics, key=lambda x: x[1], reverse=True)
#
#     # Select the top N paths (or all if n is -1)
#     top_n_paths = [path for path, metric in sorted_images[:n]]
#
#     return top_n_paths

def get_top_n_paths_per_class(basePath_hec, basePath_pc, scores_temp, metric_type,is_hec, nsample_per_class=-1):
    class_paths = defaultdict(list)

    # Group paths by class
    for path, metrics in scores_temp.items():
        class_id = path.split('/')[-2]  # Assuming the class ID is in the second-to-last part of the path

        if is_hec:  # for HEC
            path = basePath_hec + path.replace(basePath_pc, "")
        else:
            # Always the scores should be corrected to reflect Lab Workstation
            pass

        class_paths[class_id].append((path, metrics[metric_type]))



    top_n_paths = []

    # Select top N paths for each class
    for class_id, paths_metrics in class_paths.items():
        sorted_images = sorted(paths_metrics, key=lambda x: x[1], reverse=True)#should be True
        top_n_paths.extend([path for path, metric in sorted_images[:nsample_per_class]])

    return top_n_paths

# we are training on half size of all synthetic dataset
def train_syntheticAst_real(ds_name, metric_name, batch_size,trans_train,trans_val_test,is_hec):
    # ds_names = ["car_accidents", "cifar", "birds"]
    if ds_name == "birds":
        n_classes = 525
    elif ds_name == "cifar":
        n_classes = 10
    elif ds_name == "car_accidents":
        n_classes = 2
    else:
        print(f"ds_name is wrong {ds_name}")

    print("train_syntheticAst_real [0.25 0.50] less than n")
    basePath_pc = "/home/kerim/DataSets/sd2_datasets/"
    basePath_hec = "/storage/hpc/01/kerim/GANsPaper/sd2_datasets/"

    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
        score_path = '/storage/hpc/01/kerim/GANsPaper/usability_scores/scores_{}'.format(ds_name)

    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)
        score_path = '/home/kerim/PycharmProjects/SemanticSegmentation/Paper_Alpha/usability_scores/scores_{}'.format(ds_name)


    with open(score_path, "rb") as fp:  # Unpickling
        scores_temp = pickle.load(fp)

    # just for our method as we removed some synthetic examples
    n_train_r = int(len(cars_real_tr_val))

    cars_real_train, cars_real_val = random_split(cars_real_tr_val, [int(n_train_r*0.5), len(cars_real_tr_val)-int(n_train_r*0.5)])
    train_loader = []
    top_paths = []
    if metric_name=="Ours":
        # metrics_list = ["OursComp_1", "OursComp_2"]#reslts for 2 arms (on HEC folder)
        # metrics_list = ["SSIM", "PSNR", "IS", "FID"]#reslts for 4 arms (on HEC folder)
        # metrics_list = ["OursComp_1", "OursComp_2","OursComp_Avg"]# reulsts for 3 arms (on HEC)
        # metrics_list = ["OursComp_1"]  # done on HEC
        # metrics_list = ["OursComp_2"]  # done on HEC
        metrics_list = ["OursComp_1", "OursComp_2","OursComp_Avg", "Mean", "Median", "Max", "Min", "SSIM", "PSNR", "IS", "FID"]
        for metric in metrics_list:
            if ds_name =="birds":
                nsample_per_class = (len(cars_synth)//(525*2*len(metrics_list)))+8#for 11 arms
                # nsample_per_class = (len(cars_synth) // (525 * 2 * len(metrics_list)))#for 2 arms
            elif ds_name == "cifar":
                nsample_per_class = (len(cars_synth) // (10 * 2 * len(metrics_list)))
            elif ds_name == "car_accidents":
                nsample_per_class = (len(cars_synth) // (2 * 2 * len(metrics_list)))
            top_paths.append(get_top_n_paths_per_class(basePath_hec, basePath_pc, scores_temp, metric, is_hec,nsample_per_class))#//4
            print(f"I am training using our metrics on a total of {len(top_paths[0])} X {len(metrics_list)}")
    else:
        nsample_per_class = len(cars_synth) // n_classes
        nsample_per_class = nsample_per_class//2 # becaause we train on half the dataset
        top_paths = [get_top_n_paths_per_class(basePath_hec, basePath_pc,scores_temp, metric_name,is_hec, nsample_per_class)]#//should be 2 Just testing with 10
        print(f"I am training using other metrics on a total of {len(top_paths[0])}X{len(top_paths)}")
        print(top_paths[0][:9])
    # Get the indices of top paths in the dataset
    for metric_top_path in range(len(top_paths)):
        top_indices = []
        for idx, (path, _) in enumerate(cars_synth.imgs):
            if path in top_paths[metric_top_path]:
                top_indices.append(idx)
        # Create a subset containing only the top paths
        filtered_synth_train_ds = Subset(cars_synth, top_indices)
        print(f"Training on synthetic data only a pretrained model top indices {len(top_indices)} and filtered_synth_train_ds {len(filtered_synth_train_ds)}  ")

        # print(f"Synth Data Loader {metric_top_path}:{len(filtered_synth_train_ds)}, Real: {len(cars_real_train)}")

        # train_loader.append(DataLoader(ConcatDataset([filtered_synth_train_ds, cars_real_train]),  batch_size=batch_size,
        #                       shuffle=True,  num_workers=16))
        train_loader.append(DataLoader(filtered_synth_train_ds,  batch_size=batch_size,  shuffle=True,  num_workers=16))

    val_loader = DataLoader(cars_real_val,  batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(cars_real_test, batch_size=batch_size, shuffle=True, num_workers=16)

    return train_loader, val_loader, test_loader


# ['/home/kerim/DataSets/sd2_datasets/cifar/bird/image014_9.jpg', '/home/kerim/DataSets/sd2_datasets/cifar/bird/image374_6.jpg', '/home/kerim/DataSets/sd2_datasets/cifar/bird/image044_3.jpg', '/home/kerim/DataSets/sd2_datasets/cifar/bird/image362_1.jpg', '/home/kerim/DataSets/sd2_datasets/cifar/bird/image233_4.jpg', '/home/kerim/DataSets/sd2_datasets/cifar/bird/image026_5.jpg', '/home/kerim/DataSets/sd2_datasets/cifar/bird/image107_5.jpg', '/home/kerim/DataSets/sd2_datasets/cifar/bird/image385_8.jpg', '/home/kerim/DataSets/sd2_datasets/cifar/bird/image185_5.jpg']
# 0.5 + 0.5
def train_syntheticAst_real2(ds_name, metric_name, batch_size,trans_train,trans_val_test,is_hec):

    exit()
    print("train_syntheticAst_real2 [0.50  0.50] equal n")
    basePath_pc = "/home/kerim/DataSets/sd2_datasets/"
    basePath_hec = "/storage/hpc/01/kerim/GANsPaper/sd2_datasets/"
    # /storage/hpc/01/kerim/GANsPaper/sd2_datasets/car_accidents/Acc/image000.jpg
    if is_hec:
        cars_synth = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/test'.format(ds_name), transform=trans_val_test)
        score_path = '/storage/hpc/01/kerim/GANsPaper/usability_scores/{}/scores_{}_{}'.format(ds_name,ds_name,metric_name)

    else:
        cars_synth = ImageFolder(root='/home/kerim/DataSets/sd2_datasets/{}'.format(ds_name), transform=trans_train)
        cars_real_tr_val = ImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test)
        cars_real_test = ImageFolder(root='/home/kerim/DataSets/{}/test'.format(ds_name), transform=trans_val_test)
        score_path = '/home/kerim/PycharmProjects/SemanticSegmentation/Paper_Alpha/usability_scores' \
                     '/{}/scores_{}_{}'.format(ds_name,ds_name,metric_name)


    with open(score_path, "rb") as fp:  # Unpickling
        scores_temp = pickle.load(fp)
    # '/home/kerim/DataSets/sd2_datasets/car_accidents/Acc/image262.jpg should be there
    #/home/kerim/DataSets/sd2_datasets/car_accidents/Nat/image090.jpg REMOVE!!!
    if metric_name == "Ours" or metric_name == "OursV2":# We have a special way for removing outliers
        outliers_n_rem = len(scores_temp[0]) // 2  # 2 per class that was used for 0.25n
        scores = scores_temp
        for cls in scores.keys():
            cls_paths = [k for (k, v) in scores[cls]]
            cls_paths.reverse()
            cls_paths = cls_paths[:outliers_n_rem]

            for image_path in cls_paths:
                if is_hec:#for HEC
                    cars_synth.imgs.remove((basePath_hec +image_path.replace(basePath_pc,""),cls))
                else:
                    #Always the scores should be corrected to reflect Lab Workstation
                    cars_synth.imgs.remove((image_path, cls))

    else:# for all other metrics
        # give me the entries to be removed
        scores_temp[0] = scores_temp[0][:len(scores_temp[0]) // 2]
        scores = {}
        # Separate the two classes into sub-dictionaries

        for class_id, class_name in enumerate(cars_synth.class_to_idx.keys()):
            # Filter items that contain "/Nat/" in their paths
            scores[class_id] = [(path, score) for items in scores_temp.values() for path, score in items if
                                '/'+class_name+'/' in path]

        for cls in scores.keys():

            cls_paths = [k for (k, v) in scores[cls]]

            for image_path in cls_paths:
                if is_hec:#for HEC
                    #'/storage/hpc/01/kerim/GANsPaper/sd2_datasets/car_accidents/Acc/image000.jpg', 0)
                    print(image_path)
                    print(basePath_hec +image_path.replace(basePath_pc,"/"),cars_synth.class_to_idx[image_path.replace(basePath_pc,"").split('/')[1]])
                    print(cars_synth.class_to_idx, cars_synth.imgs[0])
                    # print(basePath_hec+image_path.replace(basePath_pc,""))
                    # cars_synth.imgs.remove((basePath_hec +image_path.replace(basePath_pc,""), cars_synth.class_to_idx[basePath_hec+image_path.replace(basePath_pc,"").split('/')[1]]))
                    cars_synth.imgs.remove((basePath_hec +image_path.replace(basePath_pc,""), cars_synth.class_to_idx[image_path.replace(basePath_pc,"").split('/')[1]]))
                else:
                    #Always the scores should be corrected to reflect Lab Workstation
                    cars_synth.imgs.remove((image_path, cars_synth.class_to_idx[image_path.replace(basePath_pc,"").split('/')[1]]))

    # because we will combine samples from synth and real
    # n_train_s = int(len(cars_synth) * 0.90)
    # just for our method as we removed some synthetic examples
    n_train_r = int(len(cars_real_tr_val) * 0.90)

    cars_real_train, cars_real_val = random_split(cars_real_tr_val, [int(n_train_r*0.5), len(cars_real_tr_val)-int(n_train_r*0.5)])
    # cars_synth_train = Subset(cars_synth, range(n_train_s))
    cars_synth_train = Subset(cars_synth, range(int(len(cars_real_train))))
    # cars_synth_train, cars_synth_val = random_split(cars_synth, [int(n_train_s), len(cars_synth)-int(n_train_s)])
    print(len(cars_synth_train), len(cars_real_train))
    train_loader = DataLoader(ConcatDataset([cars_synth_train, cars_real_train]),  batch_size=batch_size,
                              shuffle=True,  num_workers=16)
    # train_loader = DataLoader(cars_real_train,  batch_size=batch_size,  shuffle=True,  num_workers=8)
    val_loader = DataLoader(cars_real_val,  batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(cars_real_test, batch_size=batch_size, shuffle=True, num_workers=16)

    return train_loader, val_loader, test_loader
