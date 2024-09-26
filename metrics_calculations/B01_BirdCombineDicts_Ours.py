import json
import os
import pickle


def update_keys(scores):
    for class_id in scores.keys():
        updated_list = []
        curr_list = scores[class_id]

        for (path, value) in curr_list:
            updated_list.append((path.replace("/storage/hpc/01/kerim/GANsPaper/sd2_datasets/","/home/kerim/DataSets/sd2_datasets/"), value))
        scores[class_id] = updated_list
# Initialize an empty dictionary to store the combined data
combined_dict = {}

# Define a list of filenames for your 51 dictionaries
folder_path = "/home/kerim/PycharmProjects/SemanticSegmentation/Paper_Alpha/usability_scores/birds/birds_OURS_dicts/"
final_birds_score = "/home/kerim/PycharmProjects/SemanticSegmentation/Paper_Alpha/usability_scores/birds/scores_birds_OURS_Q"

dictionary_filenames  = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


# Load each dictionary and merge it into the combined_dict
for filename in dictionary_filenames:
    with open(folder_path+filename, "rb") as fp:  # Unpickling
        scores = pickle.load(fp)
        print(len(scores))
        update_keys(scores)
        combined_dict.update(scores)


# Save the combined dictionary as a JSON file
with open(final_birds_score, "wb") as fp:  # Pickling
    pickle.dump(combined_dict, fp)

print(f'Combined dictionary saved to  {final_birds_score}')
