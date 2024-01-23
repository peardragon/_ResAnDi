import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


def _preprocessing(ds, batch_size=32, shuffle=True, scale="minmax"):
    """
    Preprocesses a dataset for training.

    Args:
        ds (tuple): A tuple containing data and labels.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        scale (str, optional): Scaling method ("minmax" or None). Defaults to "minmax".

    Returns:
        DataLoader: A DataLoader object for the preprocessed data.
    """
    # Extract data and labels
    data = ds[0]
    labels = ds[1]

    # Get the number of samples
    n_ds = len(data)

    # Initialize an array for preprocessed data
    X = np.zeros((n_ds, 2, 1000))
    y = np.array(labels, dtype=int)

    # Default sequence length
    default_len = 1000

    # Loop through the dataset and preprocess each sample
    for idx in range(n_ds):
        traj = data[idx]
        length = traj.shape[-1]
        padded_len = default_len - length
        traj = np.pad(traj, ((0, 0), (0, 0), (padded_len, 0)))
        X[idx] = traj[0]

    # Scale the data if specified
    if scale == "minmax":
        scaler = MinMaxScaler()
        X[:, 0, :] = scaler.fit_transform(X[:, 0, :].T).T
        X[:, 1, :] = scaler.fit_transform(X[:, 1, :].T).T

    # Convert to PyTorch tensors
    X = torch.Tensor(X).unsqueeze(dim=2).type(torch.FloatTensor)
    y = torch.Tensor(y).type(torch.int64)

    # Create a PyTorch TensorDataset and DataLoader
    ds = TensorDataset(X, y)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def _preprocessing_traj(traj):
    # input shape ~ 1, 2, 112
    length = traj.shape[-1]
    default_len = 1000
    padded_len = default_len-length
    traj = np.pad(traj, ((0, 0), (0, 0), (padded_len, 0)))
    scaler = MinMaxScaler()

    traj[:, 0, :] = scaler.fit_transform(traj[:, 0, :].T).T
    traj[:, 1, :] = scaler.fit_transform(traj[:, 1, :].T).T

    return traj
