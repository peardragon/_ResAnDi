import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from numba import jit
from utils._preprocessing import _preprocessing
from utils._model_results import _get_model_results
from time import time
from itertools import groupby
from operator import itemgetter


@jit
def erasing(x, y, lower_bound, upper_bound, zero_masking):
    if zero_masking:
        zeros = np.zeros((upper_bound - lower_bound))
        x[lower_bound:upper_bound] = zeros
        y[lower_bound:upper_bound] = zeros

    return x, y


def Xy_to_dataset(X, y):
    X = X.reshape(-1, 1, 2, 1000)
    return [X, y]


def dataset_to_Xy(ds):
    ds = _preprocessing(ds, batch_size=len(ds[0]), shuffle=False)
    for X, y in ds:
        print("Dataset structure converted")
    X = X.reshape(len(X), -1).numpy()
    y = y.numpy()
    return X, y


def _get_datasets_trajwise(X, y, gradcam, scales, p1, p2):
    y = y.flatten()
    X_dataset = np.tile(X, (len(scales), 1, 1))
    X_noised_dataset = []
    X_random_dataset = []

    for i in tqdm(range(len(X))):
        padded_length = 0
        curr_attribution = gradcam[i]
        t1 = np.percentile(curr_attribution, p1)
        t2 = np.percentile(curr_attribution, p2)
        activated = np.argwhere((curr_attribution > t1) & (t2 > curr_attribution)).flatten()
        ranges = []
        X_noised = []
        X_random = []
        for k, g in groupby(enumerate(activated), lambda x: x[1] - x[0]):
            group = list(map(itemgetter(1), g))
            ranges.append((group[0], group[-1] + 1))

        for _ in scales:
            x_noised = np.copy(X[i][:1000])
            y_noised = np.copy(X[i][1000:])

            x_random = np.copy(X[i][:1000])
            y_random = np.copy(X[i][1000:])

            for range_set in ranges:
                range_length = range_set[1] - range_set[0]
                if 1000 - range_length <= 0:
                    continue
                lower_bound = np.random.randint(padded_length, 1000 - range_length)
                upper_bound = lower_bound + range_length

                x_random, y_random = erasing(x_random, y_random, lower_bound, upper_bound)

                lower_bound = range_set[0]
                upper_bound = range_set[1]

                x_noised, y_noised = erasing(x_noised, y_noised, lower_bound, upper_bound)

            X_noised.append(np.concatenate((x_noised, y_noised)))
            X_random.append(np.concatenate((x_random, y_random)))

        X_noised_dataset.append(np.array(X_noised))
        X_random_dataset.append(np.array(X_random))

    X_noised_dataset = np.array(X_noised_dataset).astype(np.float32)
    X_random_dataset = np.array(X_random_dataset).astype(np.float32)

    X_noised_dataset = np.swapaxes(X_noised_dataset, 0, 1)
    X_random_dataset = np.swapaxes(X_random_dataset, 0, 1)

    return X_dataset, X_noised_dataset, X_random_dataset, y


def _gradcam_occlusion_results_trajwise(dataset, gradcam, tag="", p1=80, p2=100):
    scale = np.linspace(0.0, 2, 1)
    tag = tag + "zero"

    if os.path.exists(f"./erasing_method/analysis_results/p{p1}_{p2}_random_total.npy"):
        erasing_total = np.load(f"./erasing_method/analysis_results/p{p1}_{p2}_erasing_total.npy")
        random_total = np.load(f"./erasing_method/analysis_results/p{p1}_{p2}_random_total.npy")
    else:
        model_name = "resnet18_8_b64_lr0.0001"
        print("Current Model:", model_name)
        erasing_total = []
        random_total = []
        noised = []
        rand = []
        results_total = []

        scales = np.tile(scale, 1)
        start = time()

        X, y = dataset_to_Xy(dataset)
        X_ds, X_noised_ds, X_random_ds, y = _get_datasets_trajwise(X, y, dataset, gradcam, scales=scales,
                                                                   model_name=model_name,
                                                                   p1=p1, p2=p2)
        del X
        print("Dataset configuration DONE")

        for index in tqdm(range(len(X_ds))):
            X, X_noised, X_random = X_ds[index], X_noised_ds[index], X_random_ds[index]

            noised_ds = Xy_to_dataset(X_noised, y)
            rand_ds = Xy_to_dataset(X_random, y)

            del X, X_noised, X_random

            noised_results, noised_acc, gt_truth = _get_model_results(noised_ds, model_name)
            noised_acc = accuracy_score(noised_results, gt_truth)
            random_results, rand_acc, gt_truth = _get_model_results(rand_ds, model_name)
            rand_acc = accuracy_score(random_results, gt_truth)

            noised.append(noised_acc)
            rand.append(rand_acc)

            results_total.append(np.array([noised_results, random_results, gt_truth]))

        erasing_total.append(noised)
        random_total.append(rand)

        np.save(f"./analysis_results/erasing_method/p{p1}_{p2}_erasing_total.npy", np.array(erasing_total))
        np.save(f"./analysis_results/erasing_method/p{p1}_{p2}_random_total.npy", np.array(random_total))

        print("Time:", time() - start)

    return erasing_total, random_total


def load_data(p1s, p2s, folder_path, filename_pattern):
    data = []
    for p1, p2 in zip(p1s, p2s):
        try:
            data.append(np.load(os.path.join(folder_path, filename_pattern.format(p1, p2))))
        except FileNotFoundError:
            print(f"No data found for p1={p1}, p2={p2}. Quitting...")
            return None
    return data


def plot_erasing_results(random_results, erasing_results):
    plt.hlines(np.mean(np.array(random_results).flatten()) * 100, xmin=0, xmax=9, label="Random", ls="--",
               color="gray")
    plt.plot(np.array(erasing_results).flatten() * 100, label="Grad-CAM based", color="k")
    plt.ylabel(r"Accuracy (%)")
    plt.xlabel("Percentile")
    plt.xticks(np.arange(10), labels=[10*(i + 1) for i in range(10)])
    plt.grid(False)
    plt.legend()
    plt.savefig("./figures/erasing_method_results.pdf")
    plt.show()


def _erasing_method_vis():
    p1s = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    p2s = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    folder_path = "./analysis_results/erasing_method"

    random_results = load_data(p1s, p2s, folder_path, "p{}_{}_random_total.npy")
    erasing_results = load_data(p1s, p2s, folder_path, "p{}_{}_erasing_total.npy")

    if random_results is None or erasing_results is None:
        return

    plot_erasing_results(random_results, erasing_results)
