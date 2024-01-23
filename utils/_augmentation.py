import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from utils._model_results import _get_model_results
import matplotlib as mpl

from utils._noising import _dataset_noising


############################################################################################################


def random_rotation_matrix():
    # Generate a random angle between 0 and 2*pi (360 degrees)
    angle = np.random.rand() * 2 * np.pi

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return rotation_matrix


def apply_rotation(trajectory_data, rotation_matrix):
    # Apply the rotation to the trajectory data, shape (2,1000)
    rotated_trajectory_data = np.dot(rotation_matrix, trajectory_data)
    return rotated_trajectory_data


def _get_rot_aug_dataset(model_wise_traj, model_wise_label, model_wise_exp, model_wise_feat, percentile):
    feat = np.mean(model_wise_feat, axis=-1)

    # Select the top n% of trajectories based on the feature
    feat_idx = np.argsort(feat)[::-1][:int(percentile * 100)]
    # Randomly select additional trajectories to have a total of 10,000 datasets

    feat_idx_add = np.random.choice(feat_idx, 10000)

    feat_rotated_traj = [
        ([apply_rotation(traj[0], random_rotation_matrix())])
        for traj in model_wise_traj[feat_idx_add]]

    feat_aug_rot = np.concatenate([np.stack(model_wise_traj), np.array(feat_rotated_traj)], axis=0)
    feat_aug_rot_label = np.concatenate([np.array(model_wise_label), np.array(model_wise_label[feat_idx_add])])
    feat_aug_rot_exp = np.concatenate([np.array(model_wise_exp), np.array(model_wise_exp[feat_idx_add])])

    return feat_aug_rot, feat_aug_rot_label, feat_aug_rot_exp


def _save_aug_dataset_rot(dataset_num, percentile):
    path = f"./dataset/augmentation/{dataset_num}_p{percentile}.npy"
    if os.path.exists(path):
        return
    dataset = np.load(f"./dataset/train_1000/{dataset_num}.npy", allow_pickle=True)
    feat = np.load(f"./backups/gradcam/GradCAM-Residual-1000_train_{dataset_num}.npy")

    traj_dataset = dataset[0]
    label_dataset = dataset[1]
    exp_dataset = dataset[2]

    aug_traj_dataset = []
    aug_label_dataset = []
    aug_exp_dataset = []

    for i in range(8):
        model_wise_traj = traj_dataset[i * 10000:(i + 1) * 10000]
        model_wise_label = label_dataset[i * 10000:(i + 1) * 10000]
        model_wise_exp = exp_dataset[i * 10000:(i + 1) * 10000]
        model_wise_feat = feat[i * 10000:(i + 1) * 10000]
        feat_aug_rot, feat_aug_rot_label, feat_aug_rot_exp = _get_rot_aug_dataset(model_wise_traj,
                                                                                  model_wise_label,
                                                                                  model_wise_exp, model_wise_feat,
                                                                                  percentile)
        aug_traj_dataset.extend(feat_aug_rot)
        aug_label_dataset.extend(feat_aug_rot_label)
        aug_exp_dataset.extend(feat_aug_rot_exp)

    aug_dataset = [aug_traj_dataset, aug_label_dataset, aug_exp_dataset]
    np.save(path, np.array(aug_dataset, dtype=object))

    return


#############################################################################################################


def _save_augmentation_results():
    for p in [60, 100]:
        total_table_path = f"./analysis_results/augmentation_results/p{p}_augmentation_results.csv"
    if os.path.exists(total_table_path):
        total_table = pd.read_csv(total_table_path)
    else:
        test_dataset = np.load(f"./dataset/test_1000/0.npy", allow_pickle=True)
        acc_total_dict = {'type': [], "acc": [], "noise": [], "opt": []}
        for noise in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            noised_dataset = _dataset_noising(test_dataset, noise)
            for feat in ["gc"]:
                for opt in ["mean"]:
                    for i in range(20):
                        BATCH_SIZE = 64
                        lr = 0.0001
                        model_name = f"resnet18_8_b{BATCH_SIZE}_lr{lr}_1000_augmentation_{feat}_{opt}_{p}"
                        model_results, ground_truth = _get_model_results(dataset=noised_dataset, model_name=model_name,
                                                                         tag=i)
                        acc_total = accuracy_score(model_results, ground_truth)
                        # for acc in acc_total:
                        acc_total_dict["type"].append(feat)
                        acc_total_dict["acc"].append(acc_total)
                        acc_total_dict["noise"].append(noise)
                        acc_total_dict["opt"].append(opt)
            total_table = pd.DataFrame(acc_total_dict)
        total_table.to_csv(total_table_path)

    return total_table


def _augmentation_results():
    merged_table = {'type': [], "acc": [], "noise": []}
    for p in [60, 100]:
        total_table_path = f"./analysis_results/augmentation_results/p{p}_augmentation_results.csv"
        if os.path.exists(total_table_path):
            total_table = pd.read_csv(total_table_path)
            for row in total_table.values:
                # print(row, row[2], row[3])
                merged_table['type'].append(p)
                merged_table['acc'].append(row[2])
                merged_table['noise'].append(row[3])
    merged_table = pd.DataFrame(merged_table)
    compare_table = merged_table[merged_table["type"].isin([60, 100])]
    compare_table["model"] = compare_table["type"]
    compare_table["model"] = compare_table["model"].replace(60, "Grad-CAM based augmentation")
    compare_table["model"] = compare_table["model"].replace(100, "Random augmentation")
    compare_table["acc"] = compare_table["acc"]*100
    ax = sns.lineplot(data=compare_table, x="noise", y="acc", hue="model", errorbar="se", palette=['k', 'gray'],
                      style="model")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Noise Scale")
    ax.legend()
    plt.savefig("./figures/augmentation_results.pdf")
    plt.show()
