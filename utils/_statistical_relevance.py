import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

from utils._preprocessing import _preprocessing_traj
from utils._resnet import resnet18_8
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

label_list = ["SubATTM", "SubCTRW", "SubFBM", "SubSBM", "SupFBM", "SupLW", "SupSBM", "BM"]

def _get_conv_response():
    model = resnet18_8()
    model.load_state_dict(torch.load("./model_backups/resnet18_8_b64_lr0.0001_1000/checkpoint.pt"), strict=False)
    model.eval()

    torch.zeros(1, 2, 1, 1000)
    feature_map_results = []
    model = resnet18_8()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    input = torch.zeros(1, 2, 1, 1000)
    with torch.no_grad():
        model.layer4.register_forward_hook(get_activation('l4'))
        output = model(input)
        baseline = activation['l4'].squeeze()

    input = torch.zeros(1, 2, 1, 1000)
    feature_map_results = []

    for i in range(1000):
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        input = torch.zeros(1, 2, 1, 1000)
        input[0][0][0][i] = 1
        input[0][1][0][i] = 1
        with torch.no_grad():
            model.layer4.register_forward_hook(get_activation('l4'))
            output = model(input)
            feature_map_results.append((activation['l4'].squeeze() - baseline).mean(0))

    responds = [[res[i] for res in feature_map_results] for i in range(32)]

    plt.plot(responds[0], label="0", c="tab:red")
    plt.plot(responds[-1], label="31", c="tab:green")
    plt.plot(responds[16], label="16", c="tab:blue")
    plt.hlines(0.9, xmin=0, xmax=1000-1, ls="--",
               color="gray")
    plt.legend(loc="upper right")

    plt.savefig("./figures/layer_response.svg")
    plt.show()


def array_split(traj):
    # traj shape 1,2,1000
    res = []
    for i in range(32):
        str = 0 * i + 25 * i
        end = 225 + 25 * i
        res.append(traj[0, :, str:end])
        # print(traj[0,:,str:end].shape)
    return res


def get_features(traj, opt):
    # Calculate mean auto-correlation
    if opt == 17:
        features = [np.mean([pd.Series(np.diff(axis, axis=-1)).autocorr(1) for axis in split]) for split in
                    array_split(traj)]

    # Calculate local kurtosis
    elif opt == 28:
        # ~ kurtosis / split 4 / mean consider over x,y
        features = [np.mean([stats.kurtosis(np.diff(split)[:, :56], axis=-1),
                             stats.kurtosis(np.diff(split)[:, 56:112], axis=-1),
                             stats.kurtosis(np.diff(split)[:, 112:112 + 56], axis=-1),
                             stats.kurtosis(np.diff(split)[:, 112 + 56:224], axis=-1)])
                    for split in array_split(traj)]

    # Calculate metric relate with singular event
    elif opt == 31:
        features = [np.max(np.std(
            (np.abs(np.diff(split, axis=-1)[:, 1:]) + 1e-6) / (np.abs(np.diff(split, axis=-1)[:, :-1]) + 1e-6),
            axis=-1))
            for split in array_split(traj)]

    # Calculate variance of velocity w/ normalize
    elif opt == 52:
        # ~ variance velocity / split 4 / mean consider over x,y
        features = [np.mean([
            np.mean((- np.std(np.diff(split)[:, :56], axis=-1)
                     + np.std(np.diff(split)[:, 56:112], axis=-1))),
            np.mean((- np.std(np.diff(split)[:, 56:112], axis=-1)
                     + np.std(np.diff(split)[:, 112:112 + 56], axis=-1))),
            np.mean((- np.std(np.diff(split)[:, 112:112 + 56], axis=-1)
                     + np.std(np.diff(split)[:, 112 + 56:224], axis=-1)))]) / np.mean(np.std(np.diff(split), axis=-1))
                    for split in array_split(traj)]

    # Calculate maximum length of same direction path
    elif opt == 53:
        def get_clear_movement_nz(traj):
            def get_max_length_info_nonzero(arr):
                max_range = 0
                final_index = 0
                curr_range = 0
                for i in range(len(arr) - 1):
                    if arr[i] == arr[i + 1] and arr[i] != 0:
                        curr_range += 1
                        if max_range < curr_range:
                            max_range = curr_range
                            final_index = i + 1
                    else:
                        curr_range = 0
                return final_index, max_range

            features = []
            traj_split = array_split(traj)
            for split in traj_split:
                x_info = get_max_length_info_nonzero(np.sign(np.diff(split[0])))
                y_info = get_max_length_info_nonzero(np.sign(np.diff(split[1])))
                if x_info[-1] < y_info[-1]:
                    features.append((np.std(np.diff(split[1])[y_info[0] - y_info[-1]:y_info[0] + 1])))
                else:
                    features.append((np.std(np.diff(split[0])[x_info[0] - x_info[-1]:x_info[0] + 1])))
            return features

        features = get_clear_movement_nz(traj)

    return features


def get_dataset_attr_values(trajs, opt):
    tot_mean_feature = []
    for traj in trajs:
        traj = _preprocessing_traj(traj)
        traj_features = get_features(traj, opt=opt)
        tot_mean_feature.append(traj_features)

    return np.array(tot_mean_feature)


def _save_dataset_feature(dataset_path, opt, save_dir):
    if os.path.exists(f"./analysis_results/statistical_relevance/features/{save_dir}.npy"):
        return 1
    traj_dataset = np.load(dataset_path, allow_pickle=True)[0]
    feats = get_dataset_attr_values(traj_dataset, opt)
    np.save(f"./analysis_results/statistical_relevance/features/{save_dir}.npy", feats)


def _get_pearson_correlation():
    # # # tot_feature_prob ~ all train dataset concat
    # n1, n2, n3, n4 means AC, CS, SG, VD feature in paper, respectably
    tot_n1 = [[] for _ in range(8)]
    tot_n2 = [[] for _ in range(8)]
    tot_n3 = [[] for _ in range(8)]
    tot_n4 = [[] for _ in range(8)]
    tot_gc = [[] for _ in range(8)]

    for num in range(5):
        n17 = np.load(
            f"./analysis_results/statistical_relevance/features/features-sliding-1000_32_interpolate_n17_train_{num}.npy",
            allow_pickle=True)
        n28 = np.load(
            f"./analysis_results/statistical_relevance/features/features-sliding-1000_32_interpolate_n28_train_{num}.npy",
            allow_pickle=True)
        n31 = np.load(
            f"./analysis_results/statistical_relevance/features/features-sliding-1000_32_interpolate_n31_train_{num}.npy",
            allow_pickle=True)
        n53 = np.load(
            f"./analysis_results/statistical_relevance/features/features-sliding-1000_32_interpolate_n53_train_{num}.npy",
            allow_pickle=True)
        n52 = np.load(
            f"./analysis_results/statistical_relevance/features/features-sliding-1000_32_interpolate_n52_train_{num}.npy",
            allow_pickle=True)
        n28_sc = np.abs(n28)
        n28_sc = MinMaxScaler().fit_transform(n28_sc.T).T

        n31_sc = np.abs(n31)
        n31_sc = MinMaxScaler().fit_transform(n31_sc.T).T

        n53_sc = np.abs(n53)
        n53_sc = MinMaxScaler().fit_transform(n53_sc.T).T

        gc = np.load(f"./Grad-CAM/GradCAM-raw-Residual-1000_train_{num}.npy", allow_pickle=True)

        for i in range(8):
            ###

            tot_n1[i].append(n17[i * 10000:(i + 1) * 10000])
            tot_n2[i].append(n17[i * 10000:(i + 1) * 10000] * n53_sc[i * 10000:(i + 1) * 10000])
            tot_n3[i].append(n31_sc[i * 10000:(i + 1) * 10000] * n28_sc[i * 10000:(i + 1) * 10000])
            tot_n4[i].append(n52[i * 10000:(i + 1) * 10000] * n28_sc[i * 10000:(i + 1) * 10000])
            tot_gc[i].append(gc[i * 10000:(i + 1) * 10000])

    pearson_table = {"label": [], "opts": [], "pearsonr": [], "spearman": []}
    for label in range(8):
        concat_gc_raw = np.concatenate(np.array(tot_gc[label]), axis=0).flatten()
        for n, tot_feat in enumerate([tot_n1, tot_n2, tot_n3, tot_n4]):
            concat_feat = np.concatenate(np.array(tot_feat[label]), axis=0).flatten()
            print(concat_feat.shape, concat_gc_raw.shape, n)
            index = ~np.isnan(concat_feat)
            concat_feat = concat_feat[index]
            concat_gc = concat_gc_raw[index]
            pearson_r = pearsonr(concat_gc, concat_feat)
            # spearman_r = spearmanr(concat_gc, concat_feat)

            pearson_table["label"].append(label_list[label])
            pearson_table["opts"].append(n)
            pearson_table["pearsonr"].append(pearson_r[0])
            # pearson_table["spearman"].append(spearman_r[0])
    np.save("./analysis_results/statistical_relevance/correlation_table.npy", pearson_table)


def _statistical_relevance_results(table_name=None, save_tag=""):

    if table_name is None:
        pearson_table = np.load("./analysis_results/statistical_relevance/correlation_table.npy", allow_pickle=True)
    else:
        pearson_table = np.load(f"./analysis_results/statistical_relevance/noised_correlation/{table_name}.npy", allow_pickle=True)

    pd_pearson = pd.DataFrame(dict(pearson_table.tolist()))
    heatmap_data = pd_pearson.pivot(columns='label', index='opts', values='pearsonr')

    # Define the colormap with a wide plateau around 0
    colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 0, 0)]  # Blue to Gray to White to Gray to Red
    n_bins = 100  # Number of bins
    cmap_name = "custom_cmap"
    color_stops = [0.0, 0.85 / 2, 0.5, 1.15 / 2, 1.0]
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(color_stops, colors)), N=n_bins)
    norm = mcolors.Normalize(vmin=-0.4, vmax=0.4)

    ax = sns.heatmap(heatmap_data, annot=True, vmin=-0.4, vmax=0.4, cmap=custom_cmap, norm=norm, linewidths=0.1,
                linecolor='white', fmt='.2f', cbar=True, annot_kws={"size": 6})
    # plt.title("Grad-CAM pearson correlation with statistical property")
    ax.set_xticklabels(["BM", "SubATTM", "SubCTRW", "SubFBM", "SubSBM", "SupFBM", "SupLW", "SupSBM"])
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal", horizontalalignment="center")
    ax.set_yticklabels(["AC", "CS", "SG", "VD"])
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(f"./figures/statistical_relevance{save_tag}.svg")
    plt.show()
