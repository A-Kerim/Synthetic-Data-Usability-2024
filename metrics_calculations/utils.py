import copy
import multiprocessing
import os
import random

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchmetrics.image.fid import FID
from torchmetrics.image.inception import IS
from torchmetrics.image import SSIM
from torchmetrics.image import PSNR
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

ssim = SSIM(data_range=1.0)
psnr = PSNR()
# import torchmetrics
# print(torchmetrics.__version__)
print(multiprocessing.cpu_count())
is_hec = "/home/kerim/PycharmProjects/SemanticSegmentation" not in os.getcwd()
trans_val_test = transforms.Compose([
    transforms.Resize([32, 32]),
    # transforms.CenterCrop([img_size, img_size]),
    transforms.ToTensor(), ])


# Load the pre-trained VGG model
vgg_model = models.vgg16(pretrained=True)
# Remove the classification layer (fc) at the end
vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-1])
# Set the model to evaluation mode
vgg_model.eval()

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomImageFolder(ImageFolder):
    def __init__(self, *args, run_mode=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_mode = run_mode

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)

        if self.run_mode == 1:
            return sample, target
        else:
            path = self.samples[index][0]
            return sample, target, path


def load_datasets(ds_name):
    if ds_name == "car_accidents":
        num_classes = 2
    elif ds_name =="cifar":
        num_classes = 10
    elif ds_name == "birds":
        num_classes = 525
    else:
       print("Error in num of classes!!")

    if is_hec:
        ds_as = CustomImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets_nonphoto/{}'.format(ds_name), transform=trans_val_test, run_mode=2)
        ds_ps = CustomImageFolder(root='/storage/hpc/01/kerim/GANsPaper/sd2_datasets/{}'.format(ds_name), transform=trans_val_test, run_mode=2)
        ds_r = CustomImageFolder(root='/storage/hpc/01/kerim/GANsPaper/{}/train'.format(ds_name), transform=trans_val_test, run_mode=2)
    else:
        ds_as = CustomImageFolder(root='/home/kerim/DataSets/sd2_datasets_nonphoto/{}'.format(ds_name), transform=trans_val_test, run_mode=2)
        ds_ps = CustomImageFolder(root='/home/kerim/DataSets/sd2_datasets/{}'.format(ds_name), transform=trans_val_test, run_mode=2)
        ds_r = CustomImageFolder(root='/home/kerim/DataSets/{}/train'.format(ds_name), transform=trans_val_test, run_mode=2)

    return ds_r, ds_as, ds_ps, num_classes


# this function will return two distributions: once for class "class_id" and one for all classes except "class_id"
def build_dist(ds, class_id, n_samples_per_class, batch_size):
    fid_i = FID(feature=192)
    fid_all_except_i = FID(feature=192)
    modes = [0, 1]
    for mode in modes:
        if mode==0:#in-class diversity only consider class_id
            class_i_indices = [i for i, (_, class_idx) in enumerate(ds.samples) if class_idx == class_id]
        else: # all other classes except class_id
            class_i_indices = [i for i, (_, class_idx) in enumerate(ds.samples) if class_idx != class_id]

        random.shuffle(class_i_indices)
        class_i_indices = class_i_indices[:n_samples_per_class]

        # Create Subset datasets for each class
        ds_st = Subset(ds, class_i_indices)

        # Set up DataLoader instances for each class
        loader = DataLoader(ds_st, batch_size=batch_size, shuffle=True, num_workers=8)
        if mode == 0:#return only the loader of class i
            class_i_loader = loader

        for i, (image, label, image_path) in enumerate(loader):
            if mode == 0:  # in-class diversity only consider class_id
                fid_i.update(tensor(image*255, dtype=torch.uint8), real=True)
            else:  # all other classes except class_id
                fid_all_except_i.update(tensor(image*255, dtype=torch.uint8), real=True)


    return fid_i, fid_all_except_i, class_i_loader


def extract_features(image_paths):
    features_final = []
    for img_id, image_path in enumerate(image_paths):
        # Load and preprocess the input image
        img = Image.open(image_path).convert('RGB')
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Extract features from the image using the VGG model
        with torch.no_grad():
            features = vgg_model(img)

        # Flatten the features
        features_final.append(features.squeeze().numpy())

# Now, I will return the average of these feaatures
    return np.mean(np.array(features_final),0)

# I am just looking at images in my class (i)
def build_VGGFeatureDists(ds, class_id, n_samples_per_class, batch_size):
    modes = [0, 1]
    # I am not using mode 1 in this paper
    for mode in modes:
        if mode==0:#in-class diversity only consider class_id
            VGGFeatures_i = []
            class_i_indices = [i for i, (_, class_idx) in enumerate(ds.samples) if class_idx == class_id]
        else: # all other classes except class_id
            VGGFeatures_all_except_i = []
            class_i_indices = [i for i, (_, class_idx) in enumerate(ds.samples) if class_idx != class_id]

        random.shuffle(class_i_indices)
        class_i_indices = class_i_indices[:n_samples_per_class]

        # Create Subset datasets for each class
        ds_st = Subset(ds, class_i_indices)

        # Set up DataLoader instances for each class
        loader = DataLoader(ds_st, batch_size=batch_size, shuffle=True, num_workers=8)

        for i, (image, label, image_path) in enumerate(loader):
            if mode == 0:  # in-class diversity only consider class_id
                VGGFeatures_i.append(extract_features(image_path))

    return np.mean(np.array(VGGFeatures_i),0)

# specifically for FID metric only
def build_dist_ds(ds, n_samples_per_class, batch_size):
    fid_all = FID(feature=192)

    # Create Subset datasets for each class
    # ds_st = Subset(ds, class_i_indices)

    # Set up DataLoader for all the dataset
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)

    for i, (image, label, image_path) in enumerate(loader):
        fid_all.update(tensor(image*255, dtype=torch.uint8), real=True)

    return fid_all


def build_dist_ds_IS(ds, n_samples_per_class, batch_size):
    is_all = IS(feature=2048)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)

    for i, (image, label, image_path) in enumerate(loader):
        is_all.update(tensor(image*255, dtype=torch.uint8))
    return is_all

# our metric
def calculate_fid_score_ours_wrapper(args):
    return calculate_fid_score_ours(*args)

# FID metric only
def calculate_fid_score_FID_wrapper(args):
    return calculate_fid_score_FID(*args)

# IS metric only
def calculate_is_score_IS_wrapper(args):
    return calculate_is_score_IS(*args)


def real_ds_subset(ds_r, n_samples_per_class):
    class_id = -1
    class_i_indices = [i for i, (_, class_idx) in enumerate(ds_r.samples) if class_idx != class_id]
    random.shuffle(class_i_indices)
    class_i_indices = class_i_indices[:n_samples_per_class]

    # Create Subset datasets for each class
    ds_r = Subset(ds_r, class_i_indices)
    return ds_r

def synth_ds_subset(ds, n_samples_per_class,chunk_id):
    class_id = -1
    class_i_indices = [i for i, (_, class_idx) in enumerate(ds.samples) if class_idx != class_id]
    # random.shuffle(class_i_indices)
    # chunk_size = 1000# for CIFAR
    chunk_size = 2000# for BIRDS
    chunks = [class_i_indices[i:i + chunk_size] for i in range(0, len(class_i_indices), chunk_size)]
    # class_i_indices = class_i_indices[:n_samples_per_class]

    # Create Subset datasets for each class
    ds = Subset(ds, chunks[chunk_id])
    return ds


# our metric
def calculate_fid_score_ours(images, clss_dist_ri, clsses_dist_r_expi, clss_dist_si, clsses_dist_s_expi):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)  # Add a batch dimension

    fid = copy.deepcopy(clss_dist_ri)
    fid.update(tensor(images * 255, dtype=torch.uint8), real=False)
    scr1 = float(fid.compute())

    fid = copy.deepcopy(clsses_dist_r_expi)
    fid.update(tensor(images * 255, dtype=torch.uint8), real=False)
    scr2 = float(fid.compute())

    fid = copy.deepcopy(clss_dist_si)
    fid.update(tensor(images * 255, dtype=torch.uint8), real=False)
    scr3 = float(fid.compute())

    fid = copy.deepcopy(clsses_dist_s_expi)
    fid.update(tensor(images * 255, dtype=torch.uint8), real=False)
    scr4 = float(fid.compute())

    return (scr3 + scr4) / (scr1 + scr2)

# FID metric only
def calculate_fid_score_FID(images, ds_dist_r):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)  # Add a batch dimension

    fid = copy.deepcopy(ds_dist_r)
    fid.update(tensor(images * 255, dtype=torch.uint8), real=False)
    scr1 = float(fid.compute())
    print(scr1)
    return scr1

# IS score only
def calculate_is_score_IS(images, ds_dist_r):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)  # Add a batch dimension

    is_dist = copy.deepcopy(ds_dist_r)

    is_dist.update(tensor(images * 255, dtype=torch.uint8))
    scr1 = is_dist.compute()
    # print(scr1)
    return float(scr1[0])


def calculate_ssim_score(image_s, loader_r):
    ssim_scores = []
    for i, (image_r, labels, image_paths) in enumerate(loader_r):
        score = ssim(image_s, image_r)
        ssim_scores.append(float(score))
        # print(i)
    # print("One synth img score SSIM was calculated!")
    res = np.mean(ssim_scores)
    ssim_scores.clear()
    del ssim_scores
    del loader_r
    return res


from concurrent.futures import ThreadPoolExecutor
def calculate_psnr_score(image_s, loader_r):
    psnr_scores = []

    def compute_psnr(image_r):
        score = psnr(image_s, image_r)
        if score is not None and np.isfinite(float(score)):
            return float(score)
        else:
            return 0

    with ThreadPoolExecutor(max_workers=9) as executor:
        for batch_images_r, _, _ in loader_r:
            psnr_scores.extend(executor.map(compute_psnr, batch_images_r))

    # print(psnr_scores)
    return np.mean(psnr_scores)

# def calculate_psnr_score(image_s, loader_r):
#
#     psnr_scores = []
#     for i, (batch_images_r, labels, image_paths) in enumerate(loader_r):
#         for image_r in batch_images_r:
#             psnr_scores.append(psnr(image_s, image_r.unsqueeze(0)))
#             # print(ssim_scores)
#
#     return np.mean(psnr_scores)

def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler (KL) divergence between two distributions.

    Parameters:
    p (array-like): The first probability distribution (true distribution).
    q (array-like): The second probability distribution (approximation).

    Returns:
    float: The KL divergence between the distributions.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    # Ensure the distributions sum to 1 and avoid division by zero or log of zero
    p /= np.sum(p)
    q /= np.sum(q)
    kl_div = np.sum(np.where(p != 0, p * np.log(p / q), 0))

    return kl_div

# with dot product of VGG features
def per_img_scores_ours(clss_dist_ri, clsses_dist_r_expi, clss_dist_si, clsses_dist_s_expi, loader_si,
                          VGG_Feat_clss_dist_ri):
    fid_score = {}
    pool = multiprocessing.Pool()  # Create a pool of worker processes

    for i, (images, labels, image_paths) in enumerate(loader_si):
        args = [(img, clss_dist_ri, clsses_dist_r_expi, clss_dist_si, clsses_dist_s_expi) for img in images]
        scores = pool.map(calculate_fid_score_ours_wrapper, args)

        for img_ind, score in enumerate(scores):
            img = Image.open(image_paths[img_ind]).convert('RGB')
            img = preprocess(img)
            img = img.unsqueeze(0)  # Add batch dimension

            # Extract features from the image using the VGG model
            with torch.no_grad():
                features = vgg_model(img)

            # Flatten the features
            image_features = features.squeeze().numpy()
            features_consist = 1/kl_divergence(VGG_Feat_clss_dist_ri,image_features)# the highger the better
            fid_score[image_paths[img_ind]] = [score, features_consist]

    pool.close()
    pool.join()

    fid_score = sorted(fid_score.items(), key=lambda x: x[1], reverse=True)
    return fid_score


def per_img_scores_FID(ds_dist_r, loader_si):
    fid_score = {}
    pool = multiprocessing.Pool()  # Create a pool of worker processes

    for i, (images, labels, image_paths) in enumerate(loader_si):
        args = [(img, ds_dist_r) for img in images]
        scores = pool.map(calculate_fid_score_FID_wrapper, args)

        for img_ind, score in enumerate(scores):
            fid_score[image_paths[img_ind]] = score
            print(fid_score[image_paths[img_ind]], image_paths[img_ind])

    pool.close()
    pool.join()

    fid_score = sorted(fid_score.items(), key=lambda x: x[1])
    print(fid_score)
    return fid_score


def per_img_scores_IS(ds_dist_r, loader_si):
    is_score = {}
    pool = multiprocessing.Pool()  # Create a pool of worker processes

    for i, (images, labels, image_paths) in enumerate(loader_si):
        args = [(img, ds_dist_r) for img in images]
        scores = pool.map(calculate_is_score_IS_wrapper,args)

        for img_ind, score in enumerate(scores):
            is_score[image_paths[img_ind]] = score
            print(is_score[image_paths[img_ind]], image_paths[img_ind])

    pool.close()
    pool.join()

    fid_score = sorted(is_score.items(), key=lambda x: x[1], reverse=True)
    # print(fid_score)
    return fid_score


def per_img_scores_SSIM(ds_r, ds_ps, batch_size,n_samples_per_class):
    loader_s = DataLoader(ds_ps, batch_size=1,num_workers=0, persistent_workers=False)
    loader_r = DataLoader(ds_r, batch_size=1, shuffle=True, num_workers=0)
    ssim_score = {}

    for i, (images, labels, image_paths) in enumerate(loader_s):
        ssim_score[image_paths[0]] = calculate_ssim_score(images, loader_r)
        print(image_paths,i)
        torch.cuda.empty_cache()

    # ssim_score = sorted(ssim_score.items(), key=lambda x: x[1], reverse=True)
    # print(ssim_score)
    torch.cuda.empty_cache()

    return ssim_score


def per_img_scores_PSNR(ds_r, ds_ps, batch_size):
    loader_s = DataLoader(ds_ps, batch_size=1)
    loader_r = DataLoader(ds_r, batch_size=batch_size, shuffle=True, num_workers=20)
    psnr_score = {}

    for i, (images, labels, image_paths) in enumerate(loader_s):
        psnr_score[image_paths[0]] = calculate_psnr_score(images, loader_r)
        print(image_paths[0], psnr_score[image_paths[0]])

    psnr_score = sorted(psnr_score.items(), key=lambda x: x[1], reverse=True)
    # print(ssim_score)
    return psnr_score
