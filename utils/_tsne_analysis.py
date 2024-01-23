import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

label_list = ["SubATTM", "SubCTRW", "SubFBM", "SubSBM", "SupFBM", "SupLW", "SupSBM", "BM"]

def feature_saving(dataset, model, tag):
    save_dir = f"./features/{tag}_"
    if os.path.exists(save_dir + "ground_truth.npy"):
        print("Exist")
        return

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    # FEATURE EXTRACTION LOOP
    # placeholders
    PREDS = []
    FEATS1 = []
    FEATS2 = []
    FEATS3 = []
    FEATS4 = []
    FC = []
    model_results = []
    ground_truth = []
    # placeholder for batch features
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()

        return hook

    # REGISTER HOOK
    model.layer1.register_forward_hook(get_features('feat1'))
    model.layer2.register_forward_hook(get_features('feat2'))
    model.layer3.register_forward_hook(get_features('feat3'))
    model.layer4.register_forward_hook(get_features('feat4'))
    model.avgpool.register_forward_hook(get_features('avgpool'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # loop through batches
    with torch.no_grad():
        for (X, y) in tqdm(dataloader):
            inputs = X.to(device)
            results = model(inputs)
            results_label = torch.argmax(results, dim=-1)
            model_results.extend(torch.squeeze(results_label).tolist())
            ground_truth.extend(torch.squeeze(y).tolist())

            # add feats and preds to lists
            feat1 = avgpool(features['feat1']).cpu().numpy()
            FEATS1.append(feat1)

            feat2 = avgpool(features['feat2']).cpu().numpy()
            FEATS2.append(feat2)

            feat3 = avgpool(features['feat3']).cpu().numpy()
            FEATS3.append(feat3)

            feat4 = avgpool(features['feat4']).cpu().numpy()
            FEATS4.append(feat4)

            FC.append(features['avgpool'].cpu().numpy())

    np.save(save_dir + "feature1.npy", np.array(FEATS1))
    np.save(save_dir + "feature2.npy", np.array(FEATS2))
    np.save(save_dir + "feature3.npy", np.array(FEATS3))
    np.save(save_dir + "feature4.npy", np.array(FEATS4))
    np.save(save_dir + "fc.npy", np.array(FC))
    np.save(save_dir + "model_results.npy", np.array(model_results))
    np.save("ground_truth.npy", np.array(ground_truth))


def _tsne_results(feature_map_dir, tsne_dir, truth, preds, wrong_pred=False, save_dir="./figures/temp.svg"):
    color_dict = {0: 'C0', 1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6", 7: "C7"}
    label_dict = {key: values for key, values in enumerate(label_list)}
    legend_details = [{'label': label, 'color': f"C{int(color)}"} for color, label in enumerate(label_list)]
    mapped_values = np.vectorize(color_dict.get)(truth)
    mapped_labels = np.vectorize(label_dict.get)(truth)

    if os.path.exists(f"./analysis_results/tsne_results/{tsne_dir}.npy"):
        X_hat = np.load(f"./analysis_results/tsne_results/{tsne_dir}.npy")
    else:
        feat = np.load(f"./analysis_results/tsne_results/{feature_map_dir}.npy", allow_pickle=True).flatten()
        feat = np.concatenate(feat).squeeze()
        tsne = TSNE(n_components=2)
        X_hat = tsne.fit_transform(feat)
        np.save(f"./analysis_results/tsne_results/{tsne_dir}.npy", X_hat)

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter1 = ax.scatter(
        X_hat[:, 0], X_hat[:, 1],
        c=[d['color'] for label in mapped_labels for d in legend_details if d['label'] == label],
        s=0.5
    )

    handles = [
        plt.Line2D(
            [0], [0],
            marker='o', color='w',
            markerfacecolor=d['color'],
            markersize=10,
            label=d['label']
        ) for d in legend_details
    ]
    labels = [d['label'] for d in legend_details]

    # legend1 = ax.legend(handles=handles, labels=labels, loc="lower left", title="Classes")
    # ax.add_artist(legend1)

    if wrong_pred:
        scatter2 = ax.scatter(
            X_hat[:, 0][truth != preds], X_hat[:, 1][truth != preds],
            c='k',
            s=0.5,
            alpha=0.5,
            label="Wrong Predictions"
        )

    plt.grid(False)
    plt.axis(False)
    plt.savefig(save_dir)
    plt.show()
