import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import pickle

sample_index = [7121, 8421, 999,
                16666, 19213, 11099,
                21312, 28888, 23451,
                33333, 31132, 37817,
                43679, 44437, 46970,
                51183, 53162, 53905,
                66330, 63249, 67588]


def _draw_example_traj(dataset, gradcam, idx, ax):
    trajectory_data_x = dataset[0][idx][0][0]  # Replace this with your actual trajectory data, x axis
    # trajectory_data_y = dataset[0][idx][0][1]  # Replace this with your actual trajectory data, y axis

    attribution_values = gradcam[idx]  # Replace this with your attribution values
    attribution_values = MinMaxScaler().fit_transform(attribution_values.reshape(-1, 1)).flatten()

    cmap = cm.get_cmap('RdBu_r')
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=attribution_values.min(), vmax=attribution_values.max()))
    sm.set_array([])

    # Plot the lines with colors based on attribution values on the provided subplot (ax)
    for i in range(len(trajectory_data_x) - 1):
        ax.plot([i, i + 1], [trajectory_data_x[i], trajectory_data_x[i + 1]], c=cmap(attribution_values[i]), lw=2)
        # ax.plot([i, i + 1], [trajectory_data_y[i], trajectory_data_y[i + 1]], c=cmap(attribution_values[i]), lw=2)

    # ax.set_title(f"Trajectory {idx}")
    # ax.set_xlabel("X Axis")  # Set your x-axis label
    # ax.set_ylabel("Y Axis")  # Set your y-axis label

    return ax


def _draw_all_sample_traj():
    if os.path.exists("./dataset/test_1000/sample_traj_datasets.pickle"):
        (fig, axs) = pickle.load(open("./dataset/test_1000/sample_traj_datasets.pickle", "rb"))
    else:
        data = np.load("./dataset/test_1000/0.npy", allow_pickle=True)
        gc = np.load("./Grad-CAM/GradCAM-Residual-1000_test_0.npy")
        labels = ["Sub ATTM", "Sub CTRW", "Sub FBM", "Sub SBM", "Sup FBM", "Sup LW",
                  "Sup SBM"]
        # Define the number of rows and columns
        num_rows = 7
        num_cols = 3

        # Create a figure with subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))
        cmap = cm.get_cmap('RdBu_r')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        k = 0
        # Iterate through the sample_index
        for n, (idx, ax) in enumerate(zip(sample_index, axs.flatten())):
            _draw_example_traj(data, gc, idx, ax)
            # ax.set_title(f"Trajectory {idx}")
            # Remove x and y axis labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if (n + 1) % 3 == 1:
                ax.set_ylabel(labels[k], fontsize=20)
                k += 1

            # Remove x and y axis
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Create a colorbar
        cax = fig.add_axes([1.01, 0.15, 0.02, 0.7])  # Adjust the position and size as needed
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("Attribution Value", fontsize=20)

        # save all underlying

        pickle.dump((fig, axs), open("./dataset/test_1000/sample_traj_datasets.pickle", "wb"))


    # Adjust layout and save/show the plot
    # fig.tight_layout()
    plt.rcParams['font.size'] = 20

    plt.rcParams['figure.dpi'] = 300
    plt.savefig("./figures/subplots_example.svg")
    plt.show()


def _draw_all_example_traj():
    # Sample data for demonstration
    dataset = np.load("./dataset/test_1000/0.npy", allow_pickle=True)
    gradcam = np.load("./Grad-CAM/GradCAM-Residual-1000_test_0.npy")

    # Define the number of rows and columns for your grid
    nrows, ncols = 7, 3

    # Create a grid of subplots using GridSpec
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(nrows, ncols, figure=fig, width_ratios=[1] * ncols)

    # Define custom y-labels for each row
    row_labels = ["Sub ATTM", "Sub CTRW", "Sub FBM", "Sub SBM", "Sup FBM", "Sup LW",
                  "Sup SBM"]  # Customize these labels as needed

    # Loop through the rows of subplots
    for row in range(nrows):
        row_axes = []  # To store the subplots for this row
        shared_ax = None

        # Create subplots for the current row
        for col in range(ncols):
            idx = row * ncols + col
            idx = sample_index[idx]
            ax = fig.add_subplot(gs[row, col], sharey=shared_ax)
            row_axes.append(ax)

            trajectory_data_x = dataset[0][idx][0][0]  # Replace this with your actual data for the x-axis
            # trajectory_data_y = dataset[0][idx][0][1]  # Replace this with your actual data for the y-axis

            # Replace this with your actual attribution values
            attribution_values = gradcam[idx]  # Make sure it matches the current idx
            attribution_values = MinMaxScaler().fit_transform(attribution_values.reshape(-1, 1)).flatten()

            cmap = plt.get_cmap('RdBu_r')

            for i in range(len(trajectory_data_x) - 1):
                ax.plot([i, i + 1], [trajectory_data_x[i], trajectory_data_x[i + 1]],
                        c=cmap(attribution_values[i]), lw=2)
                # ax.plot([i, i + 1], [trajectory_data_y[i], trajectory_data_y[i + 1]],
                #         c=cmap(attribution_values[i]), lw=2)

            # Share y-axis within the row
            if shared_ax is None:
                shared_ax = ax
            else:
                ax.get_shared_y_axes().join(shared_ax, ax)

        if row == 0:
            fig.text(0.04, 0.5, row_labels[row], va='center', rotation='vertical', fontsize=12)

    # Add a colorbar to the right of the subplots
    # cbar_ax = fig.add_axes([1.3, 0.15, 0.02, 0.7])
    # cmap = cm.get_cmap('RdBu_r')
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    # cbar = plt.colorbar(sm, cax=cbar_ax, label="Attribution Value")

    plt.tight_layout()
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['figure.dpi'] = 300
    plt.savefig("./subplots_example.svg")
    plt.show()
