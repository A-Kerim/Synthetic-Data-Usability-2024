import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset name
ds_id = 2
dataset_names = ['car_accidents', 'cifar', 'birds']
dataset_name = dataset_names[ds_id]
usability_scores_path = '/home/kerim/PycharmProjects/SemanticSegmentation/Paper_Alpha/usability_scores'

# Paths to the pickle files
pickle_files = {
    'SSIM': f'{usability_scores_path}/{dataset_name}/scores_{dataset_name}_SSIM',
    'PSNR': f'{usability_scores_path}/{dataset_name}/scores_{dataset_name}_PSNR',
    'IS': f'{usability_scores_path}/{dataset_name}/scores_{dataset_name}_IS',
    'FID': f'{usability_scores_path}/{dataset_name}/scores_{dataset_name}_FID',
    'Ours': f'{usability_scores_path}/{dataset_name}/scores_{dataset_name}_Ours'
}

# Load all the scores
scores = {}
for metric, file_path in pickle_files.items():
    with open(file_path, 'rb') as f:
        scores[metric] = pickle.load(f)

def find_image_score(image_path, scores_dict):
    for class_id, image_score_list in scores_dict.items():
        for img_path, score in image_score_list:
            if img_path == image_path:
                return score
    return None  # Return None if the image path is not found

def return_scores(scores, img_path):
    SSIM = find_image_score(img_path, scores['SSIM'])
    PSNR = find_image_score(img_path, scores['PSNR'])
    IS = find_image_score(img_path, scores['IS'])
    FID = find_image_score(img_path, scores['FID'])
    Ours = find_image_score(img_path, scores['Ours'])
    return SSIM, PSNR, IS, FID, Ours

def normalize(score, min_val, max_val):
    return (score - min_val) / (max_val - min_val)

# Collect all scores
SSIMs = []
PSNRs = []
ISs = []
FIDs = []
OurssComp_1 = []
OurssComp_2 = []

for class_id, imagePaths_scores in scores['SSIM'].items():
    for (img_path, _) in imagePaths_scores:
        SSIM, PSNR, IS, FID, Ours = return_scores(scores, img_path)
        SSIMs.append(SSIM)
        PSNRs.append(PSNR)
        ISs.append(IS)
        FIDs.append(FID)
        OurssComp_1.append(Ours[0])
        OurssComp_2.append(Ours[1])
    print(f"class id is: {class_id}")

# Convert lists to PyTorch tensors
SSIMs = torch.tensor(SSIMs).to(device)
PSNRs = torch.tensor(PSNRs).to(device)
ISs = torch.tensor(ISs).to(device)
FIDs = torch.tensor(FIDs).to(device)
OurssComp_1 = torch.tensor(OurssComp_1).to(device)
OurssComp_2 = torch.tensor(OurssComp_2).to(device)

# Calculate min and max values for each metric
min_values = {
    "SSIM": torch.min(SSIMs).item(),
    "PSNR": torch.min(PSNRs).item(),
    "IS": torch.min(ISs).item(),
    "FID": torch.min(FIDs).item(),
    "OursComp_1": torch.min(OurssComp_1).item(),
    "OursComp_2": torch.min(OurssComp_2).item()
}

max_values = {
    "SSIM": torch.max(SSIMs).item(),
    "PSNR": torch.max(PSNRs).item(),
    "IS": torch.max(ISs).item(),
    "FID": torch.max(FIDs).item(),
    "OursComp_1": torch.max(OurssComp_1).item(),
    "OursComp_2": torch.max(OurssComp_2).item()
}

scores_final_dic = {}
for class_id, imagePaths_scores in scores['SSIM'].items():
    for (img_path, _) in imagePaths_scores:
        SSIM, PSNR, IS, FID, Ours = return_scores(scores, img_path)

        SSIM_norm = normalize(SSIM, min_values['SSIM'], max_values['SSIM'])
        PSNR_norm = normalize(PSNR, min_values['PSNR'], max_values['PSNR'])
        IS_norm = normalize(IS, min_values['IS'], max_values['IS'])
        FID_norm = normalize(FID, min_values['FID'], max_values['FID'])
        OursComp_1_norm = normalize(Ours[0], min_values['OursComp_1'], max_values['OursComp_1'])
        OursComp_2_norm = normalize(Ours[1], min_values['OursComp_2'], max_values['OursComp_2'])
        OursComp_Avg = (OursComp_1_norm+OursComp_2_norm)/2
        # Store the normalized scores and statistics in the final dictionary
        scores_final_dic[img_path] = {
            'SSIM': SSIM_norm,
            'PSNR': PSNR_norm,
            'IS': IS_norm,
            'FID': FID_norm,
            'OursComp_1': OursComp_1_norm,
            'OursComp_2': OursComp_2_norm,
            'OursComp_Avg': OursComp_Avg,
            'Mean': np.mean([SSIM_norm, PSNR_norm, IS_norm, FID_norm, OursComp_1_norm, OursComp_2_norm]),
            'Median': np.median([SSIM_norm, PSNR_norm, IS_norm, FID_norm, OursComp_1_norm, OursComp_2_norm]),
            'Min': np.min([SSIM_norm, PSNR_norm, IS_norm, FID_norm, OursComp_1_norm, OursComp_2_norm]),
            'Max': np.max([SSIM_norm, PSNR_norm, IS_norm, FID_norm, OursComp_1_norm, OursComp_2_norm])
        }



# Function to display top and bottom 10 images for a given metric in one figure
def display_top_bottom_images(metric, n=10):
    sorted_images = sorted(scores_final_dic.items(), key=lambda x: x[1][metric], reverse=True)
    best_images = sorted_images[:n]
    worst_images = sorted_images[-n:]

    plt.figure(figsize=(20, 10))

    for i, (img_path, metrics) in enumerate(best_images):
        img = Image.open(img_path)
        plt.subplot(2, n, i + 1)
        plt.imshow(img)
        plt.title(f"Best {metric}: {metrics[metric]:.3f}")
        plt.axis('off')

    for i, (img_path, metrics) in enumerate(worst_images):
        img = Image.open(img_path)
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(img)
        plt.title(f"Worst {metric}: {metrics[metric]:.3f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

metrics = ['SSIM', 'PSNR', 'IS', 'FID', 'OursComp_1', 'OursComp_2', 'OursComp_Avg','Mean', 'Median', 'Min', 'Max']

for metric in metrics:
    print(f"\nTop and Bottom 10 images for {metric}:")
    display_top_bottom_images(metric, n=10)

# Open the file in binary write mode
with open(usability_scores_path+f'/combined_{dataset_name}', 'wb') as f:
    # Use pickle.dump to serialize the dictionary and save it to the file
    # pickle.dump(scores_final_dic, f)
    pass