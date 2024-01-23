import numpy as np
from tqdm import tqdm


def _add_noise_scale(trajecs, max_T, scale):
    dim = 2
    # trajecs input shape : (# of dataset, 2*max_T)
    n_traj = trajecs.shape[0]
    loc_error_amplitude = np.random.choice(np.array([1]), size=n_traj).repeat(dim)
    loc_error = (np.random.randn(n_traj * dim, int(max_T)).transpose() * loc_error_amplitude).transpose() * scale
    trajecs = trajecs[:, :].reshape(trajecs.shape[0] * dim, max_T).copy()
    trajecs += loc_error

    return trajecs


def _dataset_noising(dataset, scale):
    noised_trajset = []
    traj_dataset = dataset[0]
    for i in tqdm(range(len(traj_dataset))):
        traj = traj_dataset[i].copy()
        traj = traj.flatten().reshape(1, 2000).copy()

        noised_traj = _add_noise_scale(traj, 1000, scale).copy()

        noised_trajset.append(noised_traj.reshape(1, 2, 1000))
    noised_dataset = dataset.copy()
    noised_dataset[0] = noised_trajset
    return noised_dataset
