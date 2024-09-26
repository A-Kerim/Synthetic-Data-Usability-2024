import os
import sys
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import pickle
import warnings
from torch.utils.data import DataLoader

from utils import load_datasets, build_dist, per_img_scores_ours, per_img_scores_FID, build_dist_ds, build_dist_ds_IS, \
    per_img_scores_IS, per_img_scores_SSIM, per_img_scores_PSNR, real_ds_subset, synth_ds_subset, build_VGGFeatureDists

# pip install torch-fidelity

_ = torch.manual_seed(123)

ds_id = 0
metric_id = 4# SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)


warnings.filterwarnings("ignore")

batch_size = 64#124 1024 # Adjust as needed
ds_names = ["car_accidents", "cifar", "birds"]
metrics_names = ["SSIM", "PSNR", "IS", "FID", "Ours"]

n_samples_per_class = 64#422 5000 [CIFAR 5000 fulldataset]


is_hec = "/home/kerim/PycharmProjects/SemanticSegmentation" not in os.getcwd()

ds_r, ds_as, ds_ps, num_classes = load_datasets(ds_names[ds_id])

FinalScores = {}
if len(sys.argv) > 1:
    # python R01_UsabilityScore.py 1 0 10
    ds_id = int(sys.argv[1])  # car_accidents (0), cifar (1), birds (2)
    metric_id = int(sys.argv[2])# SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    strt_cls_id = int(sys.argv[3])# For SSIM 0 to 51
    # print("classes from {} to {}".format(strt_cls_id*10,(strt_cls_id+1)*10))
else:
    strt_cls_id = 0

print("{}_{}_{}".format(ds_names[ds_id], metrics_names[metric_id], strt_cls_id))

if metric_id == 0:# [SSIM] SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    ds_ps_prt = synth_ds_subset(ds_ps, n_samples_per_class, strt_cls_id)
    FinalScores[0] = per_img_scores_SSIM(real_ds_subset(ds_r, n_samples_per_class), ds_ps_prt, batch_size, n_samples_per_class)
    # with open("/home/kerim/PycharmProjects/SemanticSegmentation/GANsPaper/T0X_AllExps/usability_scores/scores_{}_{}_{}".format(
    #                 ds_names[ds_id], metrics_names[metric_id], strt_cls_id), "wb") as fp:# Pickling
    #     pickle.dump(FinalScores, fp)
    # torch.cuda.empty_cache()
    # sys.exit()

elif metric_id == 1:# [PSNR] SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    FinalScores[0] = per_img_scores_PSNR(real_ds_subset(ds_r, n_samples_per_class), ds_ps, batch_size)

elif metric_id == 2:# [IS] SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    # all dataset combined (class 0) just for consistency
    ds_r = real_ds_subset(ds_r, n_samples_per_class)
    ds_dist_r = build_dist_ds_IS(ds_r, n_samples_per_class, batch_size)
    loader_s = DataLoader(ds_ps, batch_size=batch_size, shuffle=True, num_workers=20)
    FinalScores[0] = per_img_scores_IS(ds_dist_r, loader_s)

elif metric_id == 3:# [FID] SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    # all dataset combined (class 0) just for consistency
    ds_r = real_ds_subset(ds_r, n_samples_per_class)
    ds_dist_r = build_dist_ds(ds_r, n_samples_per_class, batch_size)
    loader_s = DataLoader(ds_ps, batch_size=batch_size, shuffle=True, num_workers=8)
    FinalScores = per_img_scores_FID(ds_dist_r, loader_s)

elif metric_id == 4:# [OursV2] SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    VGG_Feat_clss_dist_ri = {}
# ge the distributions of each class in synthetic and real datasets
#     for class_id in range(0, 2):
    for class_id in range(num_classes):
#     for class_id in range(strt_cls_id * 10, (strt_cls_id + 1) * 10):
        VGG_Feat_clss_dist_ri = build_VGGFeatureDists(ds_r, class_id, n_samples_per_class, batch_size)

        clss_dist_ri, clsses_dist_r_expi, _ = build_dist(ds_r, class_id, n_samples_per_class, batch_size)
        clss_dist_si, clsses_dist_s_expi, loader_si = build_dist(ds_ps, class_id, n_samples_per_class, batch_size)
        FinalScores[class_id] = per_img_scores_ours(clss_dist_ri, clsses_dist_r_expi, clss_dist_si, clsses_dist_s_expi,
                                                      loader_si, VGG_Feat_clss_dist_ri)
        print(FinalScores[class_id])
        print(class_id)

if metric_id == 4:# SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    if is_hec:
        with open("/storage/hpc/01/kerim/GANsPaper/usability_scores/scores_{}_{}_{}".format(ds_names[ds_id], metrics_names[metric_id], strt_cls_id), "wb") as fp:  #Pickling
           pickle.dump(FinalScores, fp)
    else:
        with open("/home/kerim/PycharmProjects/SemanticSegmentation/Paper_Alpha/usability_scores/scores_{}_{}_{}ApproveQ".format(ds_names[ds_id], metrics_names[metric_id], strt_cls_id), "wb") as fp:  # Pickling
           pickle.dump(FinalScores, fp)
else:#[All others 0,1,2, and 3] SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    if is_hec:
        with open("/storage/hpc/01/kerim/GANsPaper/usability_scores/scores_{}_{}_{}".format(ds_names[ds_id], metrics_names[metric_id],strt_cls_id), "wb") as fp:  #Pickling
           pickle.dump(FinalScores, fp)
    else:
        with open("/home/kerim/PycharmProjects/SemanticSegmentation/Paper_Alpha/usability_scores/scores_{}_{}ApproveQ".format(ds_names[ds_id], metrics_names[metric_id]), "wb") as fp:  # Pickling
           pickle.dump(FinalScores, fp)
