# %%
import numpy as np
from stochastic.processes.continuous import FractionalBrownianMotion as fbm
from stochastic.processes.continuous import BrownianMotion as stb_bm
from math import pi as pi
from scipy.special import erfcinv
import os
import inspect

def regularize(positions: np.array,  # Positions of the trajectory to regularize
               times: np.array,  # Times at which previous positions were recorded
               T: int  # Length of the output trajectory
               ) -> np.array:  # Regularized trajectory.
    '''
    Regularizes a trajectory with irregular sampling times.
    '''
    times = np.append(0, times)
    pos_r = np.zeros(T)
    for idx in range(len(times) - 1):
        pos_r[int(times[idx]):int(times[idx + 1])] = positions[idx]
    pos_r -= pos_r[0]
    return pos_r


def bm1D(T: int,  # Length of the trajecgory
         D: float,  # Diffusion coefficient
         deltaT=False  # Sampling time
         ) -> np.array:  # Brownian motion trajectory
    '''Creates a 1D Brownian motion trajectory'''
    if D < 0:
        raise ValueError('Only positive diffusion coefficients allowed.')
    if not deltaT:
        deltaT = 1
    return np.cumsum(np.sqrt(2 * D * deltaT) * np.random.randn(int(T)))


class models_theory(object):

    def __init__(self):
        '''Constructor of the class'''

    def attm(self, T, alpha, D=1):
        if D == 1:
            return self._oneD().attm(T, alpha)
        elif D == 2:
            return self._twoD().attm(T, alpha)
        elif D == 3:
            return self._threeD().attm(T, alpha)
        else:
            raise ValueError('Incorrect walk dimension')

    def sbm(self, T, alpha, D=1):
        if D == 1:
            return self._oneD().sbm(T, alpha)
        elif D == 2:
            return self._twoD().sbm(T, alpha)
        elif D == 3:
            return self._threeD().sbm(T, alpha)
        else:
            raise ValueError('Incorrect walk dimension')

    def ctrw(self, T, alpha, D=1):
        if D == 1:
            return self._oneD().ctrw(T, alpha)
        elif D == 2:
            return self._twoD().ctrw(T, alpha)
        elif D == 3:
            return self._threeD().ctrw(T, alpha)
        else:
            raise ValueError('Incorrect walk dimension')

    def fbm(self, T, alpha, D=1):
        if D == 1:
            return self._oneD().fbm(T, alpha)
        elif D == 2:
            return self._twoD().fbm(T, alpha)
        elif D == 3:
            return self._threeD().fbm(T, alpha)
        else:
            raise ValueError('Incorrect walk dimension')

    def lw(self, T, alpha, D=1):
        if D == 1:
            return self._oneD().lw(T, alpha)
        elif D == 2:
            return self._twoD().lw(T, alpha)
        elif D == 3:
            return self._threeD().lw(T, alpha)
        else:
            raise ValueError('Incorrect walk dimension')

    def standardBM(self, T, alpha, D=1):
        if D == 1:
            return self._oneD().standardBM(T, alpha)
        elif D == 2:
            return self._twoD().standardBM(T, alpha)
        elif D == 3:
            return self._threeD().standardBM(T, alpha)
        else:
            raise ValueError('Incorrect walk dimension')


class models_theory(models_theory):
    class _twoD():

        # @njit
        def ctrw(self, T, alpha, regular_time=True):
            ''' Creates a 2D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time. '''
            if alpha > 1:
                raise ValueError('Continuous random walks only allow for anomalous exponents <= 1.')
                # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = times[:np.argmax(times > T) + 1]
            # Generate the positions of the walk
            posX = np.cumsum(np.random.randn(len(times)))
            posY = np.cumsum(np.random.randn(len(times)))
            posX -= posX[0]
            posY -= posY[0]
            # Regularize and output
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T)
                return np.concatenate((regX, regY))
            else:
                return np.stack((times, posX, posY))

        # @njit
        def fbm(self, T, alpha):
            ''' Creates a 2D fractional brownian motion trajectory'''
            # Defin Hurst exponent
            H = alpha * 0.5
            return np.concatenate((fbm(hurst=H).sample(int(T - 1)), fbm(hurst=H).sample(int(T - 1))))

        # @njit
        def lw(self, T, alpha):
            ''' Creates a 2D Levy walk trajectory '''
            if alpha < 1:
                raise ValueError('Levy walks only allow for anomalous exponents > 1.')
                # Define exponents for the distribution of times
            if alpha == 2:
                sigma = np.random.rand()
            else:
                sigma = 3 - alpha
            dt = (1 - np.random.rand(T)) ** (-1 / sigma)
            dt[dt > T] = T + 1
            # Define the velocity
            v = 10 * np.random.rand()
            # Define the array where we save step length
            d = np.empty(0)
            # Define the array where we save the angle of the step
            angles = np.empty(0)
            # Generate trajectory
            for t in dt:
                d = np.append(d, v * np.ones(int(t)) * (2 * np.random.randint(0, 2) - 1))
                angles = np.append(angles, np.random.uniform(low=0, high=2 * pi) * np.ones(int(t)))
                if len(d) > T:
                    break
            d = d[:int(T)]
            angles = angles[:int(T)]
            posX, posY = [d * np.cos(angles), d * np.sin(angles)]
            return np.concatenate((np.cumsum(posX) - posX[0], np.cumsum(posY) - posY[0]))

        # @njit
        def attm(self, T, alpha, regime=1):
            '''Creates a 2D trajectory following the annealed transient time model
            Optional parameters:
                :regime (int):
                    - Defines the ATTM regime. Accepts three values: 0,1,2.'''
            if regime not in [0, 1, 2]:
                raise ValueError('ATTM has only three regimes: 0, 1 or 2.')
            if alpha > 1:
                raise ValueError('ATTM only allows for anomalous exponents <= 1.')
                # Gamma and sigma selection
            if regime == 0:
                sigma = 3 * np.random.rand()
                gamma = np.random.uniform(low=-5, high=sigma)
                if alpha < 1:
                    raise ValueError('ATTM regime 0 only allows for anomalous exponents = 1.')
            elif regime == 1:
                sigma = 3 * np.random.uniform(low=1e-2, high=1.1)
                gamma = sigma / alpha
                while sigma > gamma or gamma > sigma + 1:
                    sigma = 3 * np.random.uniform(low=1e-2, high=1.1)
                    gamma = sigma / alpha
            elif regime == 2:
                gamma = 1 / (1 - alpha)
                sigma = np.random.uniform(low=1e-2, high=gamma - 1)
            # Generate the trajectory
            posX = np.array([0])
            posY = np.array([0])
            while len(posX) < T:
                Ds = (1 - np.random.uniform(low=0.1, high=0.99)) ** (1 / sigma)
                ts = Ds ** (-gamma)
                if ts > T:
                    ts = T
                posX = np.append(posX, posX[-1] + bm1D(ts, Ds))
                posY = np.append(posY, posY[-1] + bm1D(ts, Ds))
            return np.concatenate((posX[:T] - posX[0], posY[:T] - posY[0]))

        # @njit
        def sbm(self, T, alpha, sigma=1):
            '''Creates a scaled brownian motion trajectory'''
            msd = (sigma ** 2) * np.arange(T + 1) ** alpha
            deltas = np.sqrt(msd[1:] - msd[:-1])
            dx = np.sqrt(2) * deltas * erfcinv(2 - 2 * np.random.rand(len(deltas)))
            dy = np.sqrt(2) * deltas * erfcinv(2 - 2 * np.random.rand(len(deltas)))
            return np.concatenate((np.cumsum(dx) - dx[0], np.cumsum(dy) - dy[0]))

        # @njit
        def standardBM(self, T, alpha=1):
            ''' Creates a 2D brownian motion trajectory'''
            return np.concatenate((stb_bm().sample(int(T - 1)), stb_bm().sample(int(T - 1))))






def normalize(trajs):
    '''
    Normalizes trajectories by substracting average and dividing by
    SQRT of their standard deviation.

    Parameters
    ----------
    trajs : np.array
        Array of length N x T or just T containing the ensemble or single trajectory to normalize.
    '''
    # Checking and saving initial shape
    initial_shape = trajs.shape
    if len(trajs.shape) == 1:  # single one d trajectory
        trajs = trajs.reshape(1, trajs.shape[0], 1)
    if len(trajs.shape) == 2:  # ensemble of one d trajectories
        trajs = trajs.reshape(trajs.shape[0], trajs.shape[1], 1)

    trajs = trajs - trajs.mean(axis=1, keepdims=True)
    displacements = (trajs[:, 1:, :] - trajs[:, :-1, :]).copy()
    variance = np.std(displacements, axis=1)
    variance[variance == 0] = 1
    new_trajs = np.cumsum((displacements / np.expand_dims(variance, axis=1)), axis=1)
    initial_zeros = np.expand_dims(np.zeros((new_trajs.shape[0], new_trajs.shape[-1])), axis=1)
    return np.concatenate((initial_zeros, new_trajs), axis=1).reshape(initial_shape)


# %%
class datasets_theory():

    def __init__(self):
        '''
        This class generates, saves and loads datasets of theoretical trajectories simulated
        from various diffusion models (available at andi_datasets.models_theory).
        '''
        self._dimension = 2
        self._get_models()

    def _get_models(self):
        '''Loading subclass of models'''
        if self._dimension == 1:
            self._models = models_theory._oneD()
        elif self._dimension == 2:
            self._models = models_theory._twoD()
        elif self._dimension == 3:
            self._models = models_theory._threeD()
        else:
            raise ValueError(
                'Our current understanding of the physical world is three dimensional and so are the diffusion models available in this class')

        available_models = inspect.getmembers(self._models, inspect.ismethod)
        self.avail_models_name = [x[0] for x in available_models]
        self.avail_models_func = [x[1] for x in available_models]

    def create_dataset(self, T, N_models, exponents, models,
                       dimension=1,
                       save_trajectories=False, load_trajectories=False,
                       path='datasets/',
                       N_save=1000, t_save=1000):
        '''
        Creates a dataset of trajectories via the theoretical models defined in `.models_theory`. Check our tutorials for use cases of this function.

        Parameters
        ----------
        T : int
            Length of the trajectories.
        N_models : int, numpy.array
            - if int, number of trajectories per class (i.e. exponent and model) in the dataset.
            - if numpy.array, number of trajectories per classes: size (number of models)x(number of classes)
        exponents : float, array
            Anomalous exponents to include in the dataset. Allows for two digit precision.
        models : bool, int, list
            Labels of the models to include in the dataset.
            Correspodance between models and labels is given by self.label_correspodance, defined at init.
            If int/list, choose the given models. If False, choose all of them.
        dimensions : int
            Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
        save_trajectories : bool
            If True, the module saves a .h5 file for each model considered, with N_save trajectories
            and T = T_save.
        load_trajectories : bool
            If True, the module loads the trajectories of an .h5 file.
        path : str
            Path to the folder where to save/load the trajectories dataset.
        N_save : int
            Number of trajectories to save for each exponents/model.
            Advise: save at the beggining a big dataset (t_save ~ 1e3 and N_save ~ 1e4)
            which then allows you to load any other combiantion of T and N_models.
        t_save : int
            Length of the trajectories to be saved. See comments on N_save.

        Returns
        -------
        numpy.array
                - Dataset of trajectories of lenght Nx(T+2), with the following structure:
                    o First column: model label
                    o Second column: value of the anomalous exponent
                    o 2:T columns: trajectories
        '''

        '''Managing probable errors in inputs'''
        if T < 2:
            raise ValueError('The time of the trajectories has to be bigger than 1.')
        if isinstance(exponents, int) or isinstance(exponents, float):
            exponents = [exponents]

        '''Managing folders of the datasets'''
        if save_trajectories or load_trajectories:
            if load_trajectories:
                save_trajectories = False
            if not os.path.exists(path) and load_trajectories:
                raise FileNotFoundError('The directory from where you want to load the dataset does not exist')
            if not os.path.exists(path) and save_trajectories:
                os.makedirs(path)

        '''Establish dimensions and corresponding models'''
        self._dimension = dimension
        self._get_models()

        '''Managing models to load'''
        # Load from a list of models
        if isinstance(models, list):
            self._models_name = [self.avail_models_name[idx] for idx in models]
            self._models_func = [self.avail_models_func[idx] for idx in models]
        # Load from a single model
        elif isinstance(models, int) and not isinstance(models, bool):
            self._models_name = [self.avail_models_name[models]]
            self._models_func = [self.avail_models_func[models]]
        # Load all available models
        else:
            self._models_name = self.avail_models_name
            self._models_func = self.avail_models_func

        '''Managing number of trajectory per class:
            - Defines array num_class as a function of N'''
        if isinstance(N_models, int):
            n_per_class = N_models * np.ones((len(self._models_name), len(exponents)))

        elif type(N_models).__module__ == np.__name__:
            if len(self._models_name) != N_models.shape[0] or len(exponents) != N_models.shape[1]:
                raise ValueError('Mismatch between the dimensions of N and the number of different classes.' +
                                 f'N must be either an int (balanced classes) or an array of length {len(models)}x'
                                 f'{len(exponents)} (inbalaced classes).')
            n_per_class = N_models
        else:
            raise TypeError('Type of variable N not recognized.')

        '''Defining default values for saved datasets'''
        N_save = np.ones_like(n_per_class) * N_save
        # If the number of class of a given class is bigger than N_save, we
        # change the value of N_save for that particular class.
        N_save = np.max([N_save, n_per_class], axis=0)

        data_models = self._create_trajectories(T=T,
                                                exponents=exponents,
                                                dimension=self._dimension,
                                                models_name=self._models_name,
                                                models_func=self._models_func,
                                                n_per_class=n_per_class)

        return data_models

    def _create_trajectories(self, T, exponents, dimension, models_name, models_func, n_per_class):
        ''' create a dataset for the exponents and models considered.
        Arguments:
            :T (int):
                - length of the trajectories.
            :exponents (array):
                - anomalous exponents to include in the dataset. Allows for two digit precision.
            :dimension (int):
                - Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
            :models_name (list of str):
                - names of the models to include in the output dataset.
            :models_func (list of funcs):
                - function generating the models to include in the output dataset.
            :n_per_class:
                - number of trajectories to consider per exponent/model.
        Return:
            :dataset (numpy.array):
                - Dataset of trajectories of lenght (number of models)x(T+2), with the following structure:
                    o First column: model label.
                    o Second column: value of the anomalous exponent.
                    o 2:T columns: trajectories.'''

        for idx_m, (name, func) in enumerate(zip(models_name, models_func)):
            for idx_e, exp in enumerate(exponents):

                n = int(n_per_class[idx_m, idx_e])
                data = np.zeros((n, self._dimension * T))
                for i in range(n):
                    data[i, :] = func(T, exp)

                data = self._label_trajectories(trajs=data, model_name=name, exponent=exp)

                if idx_e + idx_m == 0:
                    dataset = data
                else:
                    dataset = np.concatenate((dataset, data), axis=0)

        return dataset

    def _label_trajectories(self, trajs, model_name, exponent):
        ''' Labels given trajectories given the corresponding label for the model and exponent.
        For models, the label correspond to the position of the model in self.avail_models_name.
        For exponents, the label if the value of the exponent.
        Arguments:
            :trajs (numpy array):
                - trajectories to label
            :model_name (str):
                - name of the model from which the trajectories are coming from.
            :exponent (float):
                - Anomalous exponent of the trajectories.
        Return:
            :trajs (numpy array):
                - Labelled trajectoreis, with the following structure:
                    o First column: model label
                    o Second columnd: exponent label
                    o Rest of the array: trajectory.   '''

        label_model = self.avail_models_name.index(model_name)

        labels_mod = np.ones((trajs.shape[0], 1)) * label_model
        labels_alpha = np.ones((trajs.shape[0], 1)) * exponent
        trajs = np.concatenate((labels_mod, labels_alpha, trajs), axis=1)

        return trajs

    def create_noisy_localization_dataset(self,
                                          dataset=False,
                                          T=False, N=False, exponents=False, models=False, dimension=1,
                                          noise_func=False, sigma=1, mu=0,
                                          save_trajectories=False, load_trajectories=False,
                                          path='datasets/',
                                          N_save=1000, t_save=1000):
        '''
        Create a dataset of noisy trajectories.
        This function creates trajectories with _create_trajectories and then adds given noise to them.
        All parameters are the same as _create_trajectories but noise_func.

        Parameters
        ----------
        dataset : bool, numpy array
            If False, creates a dataset with the given parameters.
            If numpy array, dataset to which the function applies the noise.
        noise_func : bool, function
            If False, the noise added to the trajectories will be Gaussian distributed, with
            variance sigma and mean value mu.
            If function, uses the given function to generate noise to be added to the trajectory.
            The function must have as input two ints, N and M and the output must be a matrix of size NxM.

        Returns
        -------
        numpy.array
            Dataset of trajectories of lenght Nx(T+2), with the following structure:
                o First column: model label
                o Second column: value of the anomalous exponent
                o 2:T columns: trajectories'''

        if not dataset.any():
            dataset = self.create_dataset(T, N, exponents, models, dimension,
                                          save_trajectories, load_trajectories,
                                          path,
                                          N_save, t_save)

        # Add the noise to the trajectories
        trajs = dataset[:, 2:].reshape(dataset.shape[0] * dimension, T)
        trajs = self._add_noisy_localization(trajs, noise_func, sigma, mu)

        dataset[:, 2:] = trajs.reshape(dataset.shape[0], T * dimension)

        return dataset

    @staticmethod
    def _add_noisy_localization(trajs, noise_func=False, sigma=1, mu=0):

        if isinstance(noise_func, np.ndarray):
            noise_matrix = noise_func
        elif not noise_func:
            noise_matrix = sigma * np.random.randn(trajs.shape) + mu
        elif hasattr(noise_func, '__call__'):
            noise_matrix = noise_func(trajs)
        else:
            raise ValueError('noise_func has to be either False for Gaussian noise, a Python function or numpy array.')

        trajs += noise_matrix

        return trajs


def _add_noise_with_normalization_scale(trajecs, max_T, noise_scale):
    dim = 2
    n_traj = trajecs.shape[0]
    norm_trajs = normalize(trajecs[:, 2:].reshape(n_traj * dim, max_T))
    trajecs[:, 2:] = norm_trajs.reshape(trajecs[:, 2:].shape)
    loc_error_amplitude = np.random.choice(np.array([noise_scale]), size=n_traj).repeat(dim)
    # loc_error_amplitude = np.random.choice(np.array([0.1, 0.5, 1]), size = n_traj).repeat(dim)
    loc_error = (np.random.randn(n_traj * dim, int(max_T)).transpose() * loc_error_amplitude).transpose()
    trajecs = datasets_theory().create_noisy_localization_dataset(trajecs, dimension=dim, T=max_T, noise_func=loc_error)
    return trajecs



def dataset_generation(length="Fixed", noise="None", dataset_num=0, save_dir=f"./dataset/test_1000/"):
    np.random.seed(42)
    AD = datasets_theory()

    model_id = AD.avail_models_name
    for i in range(len(model_id)):
        model_id[i] = model_id[i].upper()

    # devide due to levy cannot creat sub-diffusion
    dim = 2  # 2D, 3D [model_id, exponents, x:tmax, y:tmax, z:tmax]
    twmins = list(np.ones(1) * 1000) if length == "Fixed" else list(np.ones(1) * 10)
    noise_scale = 0 if noise == "None" else [0.1, 0.5, 1]  # default

    twmax = 1001
    nmax = 10000

    for itry, twmin in (enumerate(twmins)):

        tmax = np.random.randint(low=twmin, high=twmax, size=nmax)

        sub_exponents_candi = np.random.uniform(low=0.1, high=0.9, size=nmax)
        sup_exponents_candi = np.random.uniform(low=1.1, high=1.9, size=nmax)

        sub_models = [0, 1, 2, 4]
        sup_models = [2, 3, 4]
        bm = [5]

        n_label = 0

        dataset = []
        labelset = []
        exponset = []

        for im, imod in (enumerate(sub_models)):
            icount = 0
            while icount <= nmax - 1:
                trajecs = AD.create_dataset(T=tmax[icount], N_models=1, exponents=sub_exponents_candi[icount],
                                            models=imod, dimension=dim)
                trajecs = trajecs.copy()
                norm_noised_traj = _add_noise_with_normalization_scale(trajecs=trajecs, max_T=tmax[icount],
                                                                       noise_scale=noise_scale)

                if np.sum(np.square((norm_noised_traj[0][2:]))) != 0:
                    labelset.append(np.zeros_like(norm_noised_traj[0][0]) + n_label)
                    exponset.append(norm_noised_traj[0][1])
                    dataset.append(norm_noised_traj[0][2:].reshape(-1, dim, tmax[icount]))
                    icount = icount + 1
            n_label += 1

        for im, imod in (enumerate(sup_models)):
            icount = 0
            while icount <= nmax - 1:
                trajecs = AD.create_dataset(T=tmax[icount], N_models=1, exponents=sup_exponents_candi[icount],
                                            models=imod, dimension=dim)
                trajecs = trajecs.copy()
                norm_noised_traj = _add_noise_with_normalization_scale(trajecs=trajecs, max_T=tmax[icount],
                                                                       noise_scale=noise_scale)

                if np.sum(np.square((norm_noised_traj[0][2:]))) != 0:
                    labelset.append(np.zeros_like(norm_noised_traj[0][0]) + n_label)
                    exponset.append(norm_noised_traj[0][1])
                    dataset.append(norm_noised_traj[0][2:].reshape(-1, dim, tmax[icount]))

                icount = icount + 1
            n_label += 1

        for im, imod in (enumerate(bm)):
            icount = 0
            while icount <= nmax - 1:
                trajecs = AD.create_dataset(T=tmax[icount], N_models=1, exponents=sup_exponents_candi[icount],
                                            models=imod, dimension=dim)
                trajecs = trajecs.copy()
                norm_noised_traj = _add_noise_with_normalization_scale(trajecs=trajecs, max_T=tmax[icount],
                                                                       noise_scale=noise_scale)

                if np.sum(np.square((norm_noised_traj[0][2:]))) != 0:
                    labelset.append(np.zeros_like(norm_noised_traj[0][0]) + n_label)
                    exponset.append(1)
                    dataset.append(norm_noised_traj[0][2:].reshape(-1, dim, tmax[icount]))
                icount = icount + 1
            n_label += 1

        with open(save_dir + f"{dataset_num}.npy", 'wb') as f:
            np.save(f, np.array([dataset, labelset, exponset], dtype=object))

def add_noise(load_dir, save_dir, noise_scale):
    np.random.seed(42)
    dim = 2
    nmax = 10000
    icount = 0
    n_label = 0

    trajs = np.load(load_dir, allow_pickle=True)

    if os.path.exists(save_dir):
        return
    labelset = []
    exponset = []
    dataset = []
    for trajecs in trajs:
        trajecs = _add_noise_with_normalization_scale(trajecs=np.array(trajecs), max_T=1000, noise_scale=noise_scale)

        if np.sum(np.square((trajecs[0][2:]))) != 0:
            labelset.append(np.zeros_like(trajecs[0][0]) + n_label)
            exponset.append(trajecs[0][1])  # brownian motion exp==1
            dataset.append(trajecs[0][2:].reshape(-1, dim, 1000))
            icount = icount + 1
        if icount <= nmax - 1:
            icount = 0
            n_label += 1

    with open(save_dir, 'wb') as f:
        np.save(f, np.array([dataset, labelset, exponset], dtype=object))



