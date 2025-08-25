from __future__ import annotations

import abc
import os
import pathlib
import pickle
from typing import List, Optional, Tuple


import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset
from pathlib import Path
from src.data_utils  import PAMDataset, process_MITECG, process_Boiler_WinIT, process_Synth, \
    process_Boiler_OLD, process_MITECG_for_WINIT, process_PAM, process_Epilepsy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


class WinITDataset(abc.ABC):
    """
    Dataset abstract class that needed to run using our code.
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        batch_size: int,
        testbs: int | None,
        deterministic: bool,
        cv_to_use: List[int] | int | None,
        seed: int | None,
    ):
        """
        Constructor

        Args:
            data_path:
                The path of the data.
            batch_size:
                The batch size of the train loader and valid loader.
            testbs:
                The batch size for the test loader.
            deterministic:
                Indicate whether deterministic algorithm is to be used. GLOBAL behaviour!
            cv_to_use:
                Indicate which cv to use. CV are from 0 to 4.
            seed:
                The random seed.

        """


        if isinstance(data_path, str):
            self.data_path = Path(data_path)
        else:
            self.data_path = data_path

        self.train_loaders: List[DataLoader] | None = None
        self.valid_loaders: List[DataLoader] | None = None
        self.test_loader: DataLoader | None = None
        self.feature_size: int | None = None

        self.seed = seed
        self.batch_size = batch_size
        self.testbs = testbs
        self._cv_to_use = cv_to_use
        print("self._cv_to_use:", self._cv_to_use)

        torch.set_printoptions(precision=8)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True

    def _get_loaders(
        self,
        train_data: np.ndarray,
        train_label: np.ndarray,
        test_data: np.ndarray,
        test_label: np.ndarray,
        valid_data: Optional[np.ndarray] = None,
        valid_label: Optional[np.ndarray] = None,
    ):
        """
        Get the train loader, valid loader and the test loaders. The "train_data" and "train_label"
        will be split to be the training set and the validation set.

        Args:
            train_data: (N, D, T)
                The train data
            train_label:
                The train label
            test_data: (N, D, T)
                The test data
            test_label:
                The test label
        """
        # print(train_data)
        # print(len(train_data))
        # print(train_label)
        # print(len(train_label))
        feature_size = train_data.shape[1]
        train_tensor_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
        test_tensor_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

        testbs = self.testbs if self.testbs is not None else len(test_data)

        train_loaders = []
        valid_loaders = []
        if valid_data is None:
            kf = KFold(n_splits=5)

            for train_indices, valid_indices in kf.split(train_data):
                train_subset = Subset(train_tensor_dataset, train_indices)
                valid_subset = Subset(train_tensor_dataset, valid_indices)
                train_loaders.append(DataLoader(train_subset, batch_size=self.batch_size))
                valid_loaders.append(DataLoader(valid_subset, batch_size=self.batch_size))
            
            self.train_loaders = train_loaders
            self.valid_loaders = valid_loaders

        else:
            valid_tensor_dataset = TensorDataset(torch.Tensor(valid_data), torch.Tensor(valid_label))
            train_loader = DataLoader(train_tensor_dataset, batch_size=testbs)
            valid_loader = DataLoader(valid_tensor_dataset, batch_size=testbs)
            train_loaders.append(train_loader)
            valid_loaders.append(valid_loader)
            self.train_loaders = train_loaders
            self.valid_loaders = valid_loaders
        
        test_loader = DataLoader(test_tensor_dataset, batch_size=testbs)
        self.test_loader = test_loader
        self.feature_size = feature_size

    @abc.abstractmethod
    def load_data(self) -> None:
        """
        Load the data from the file.
        """

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the dataset.
        """

    @property
    @abc.abstractmethod
    def data_type(self) -> str:
        """
        Return the type of the dataset. (Not currently used)
        """

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """
        Return the number of classes
        """

    def num_cv(self) -> int:
        """
        Return the total number of CV
        """
        if self.train_loaders is None:
            return 0
        return len(self.train_loaders)

    def cv_to_use(self) -> List[int]:
        """
        Return a list of CV to use.
        """
        if self.train_loaders is None:
            return [0]
        num_cv = self.num_cv()
        print('num_cv:', num_cv)
        print('_cv_to_use:', self._cv_to_use)
        if self._cv_to_use is None:
            return list(range(num_cv))
        if isinstance(self._cv_to_use, int) and 0 <= self._cv_to_use < num_cv:
            return [self._cv_to_use]
        if isinstance(self._cv_to_use, list) and all(0 <= c < num_cv for c in self._cv_to_use):
            return self._cv_to_use
        raise ValueError("CV to use range is invalid.")

    def get_train_loader(self, cv: int) -> DataLoader | None:
        """
        Get the train loader to the corresponding CV.
        """
        if self.train_loaders is None:
            return None
        return self.train_loaders[cv]

    def get_valid_loader(self, cv: int) -> DataLoader | None:
        """
        Get the valid loader to the corresponding CV.
        """
        if self.valid_loaders is None:
            return None
        return self.valid_loaders[cv]

    def get_test_loader(self) -> DataLoader | None:
        """
        Return the test loader.
        """
        return self.test_loader




    

class SimulatedData(WinITDataset, abc.ABC):
    """
    An abstract class for simulated data.
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        batch_size: int,
        testbs: int | None,
        deterministic: bool,
        file_name_prefix: str,
        ground_truth_prefix: str,
        cv_to_use: List[int] | int | None,
        seed: int | None,
    ):
        """
        Constructor

        Args:
            data_path:
                The path of the data.
            batch_size:
                The batch size of the train loader and valid loader.
            testbs:
                The batch size for the test loader.
            deterministic:
                Indicate whether deterministic algorithm is to be used. GLOBAL behaviour!
            file_name_prefix:
                The file name prefix for the train and the test data. The names of the files will
                be [PREFIX]x_train.pkl, [PREFIX]y_train.pkl, [PREFIX]x_test.pkl and
                [PREFIX]y_test.pkl.
            ground_truth_prefix:
                The ground truth importance file prefix. The file name will be [PREFIX]_test.pkl
            cv_to_use:
                Indicate which cv to use. CV are from 0 to 4.
            seed:
                The random seed.
        """
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name_prefix = file_name_prefix
        self.ground_truth_prefix = ground_truth_prefix

    def load_data(self, train_ratio=0.8) -> None:
        with (self.data_path / f"{self.get_name()}_data" / f"{self.file_name_prefix}x_train.pkl").open("rb") as f:
            train_data = pickle.load(f)
        with (self.data_path / f"{self.get_name()}_data" /f"{self.file_name_prefix}y_train.pkl").open("rb") as f:
            train_label = pickle.load(f)
        with (self.data_path / f"{self.get_name()}_data" /f"{self.file_name_prefix}x_test.pkl").open("rb") as f:
            test_data = pickle.load(f)
        with (self.data_path / f"{self.get_name()}_data" /f"{self.file_name_prefix}y_test.pkl").open("rb") as f:
            test_label = pickle.load(f)

        rng = np.random.default_rng(seed=self.seed)
        perm = rng.permutation(train_data.shape[0])
        train_data = train_data[perm]
        train_label = train_label[perm]

        self._get_loaders(train_data, train_label, test_data, test_label)

    @property
    def num_classes(self) -> int:
        return 1

    def load_ground_truth_importance(self) -> np.ndarray:
        with open(os.path.join(self.data_path, f"{self.get_name()}_data", self.ground_truth_prefix + "_test.pkl"), "rb") as f:
            gt = pickle.load(f)
        return gt


class SimulatedState(SimulatedData):
    """
    Simulated State data
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "state_dataset_",
        ground_truth_prefix: str = "state_dataset_importance",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        return "simulated_state"

    @property
    def data_type(self) -> str:
        return "state"

class SimulatedSwitch(SimulatedData):
    """
    Simulated Switch data
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "state_dataset_",
        ground_truth_prefix: str = "state_dataset_importance",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        return "simulated_switch"

    @property
    def data_type(self) -> str:
        return "switch"


class SimulatedSpike(SimulatedData):
    """
    Simulated Spike data, with possible delay involved.
    """

    def __init__(
        self,
        data_path: pathlib.Path = None,
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "",
        ground_truth_prefix: str = "gt",
        delay: int = 0,
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        if data_path is None:
            if delay > 0:
                data_path = pathlib.Path(f"./data/simulated_spike_data_delay_{delay}")
            elif delay == 0:
                data_path = pathlib.Path("./data/simulated_spike_data")
            else:
                raise ValueError("delay must be non-negative.")
        self.delay = delay

        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        if self.delay == 0:
            return "simulated_spike"
        return f"simulated_spike_delay_{self.delay}"

    @property
    def data_type(self) -> str:
        return "spike"
    

class SeqCombMV(SimulatedData):
    """
    SeqCombMV data, with possible delay involved.
    num_samples: 6100
    T = 200
    D = 4
    Classes = 4
    """

    def __init__(
        self,
        data_path: pathlib.Path = None,
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "",
        ground_truth_prefix: str = "gt",
        delay: int = 0,
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
        device: str  = 'cuda',
                

    ):
        if data_path is None:
            data_path = pathlib.Path(f"./data/")
        self.device = device

        self.split_cv = cv_to_use
        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )


    def load_D(self):
        """
        Lightweight loader for self.D without doing full load_data pipeline.
        Only loads the raw processed dataset into self.D.
        """
        if not hasattr(self, "D") or self.D is None:
            self.D = process_Synth(
                split_no=self.split_cv[0] + 1,
                device=self.device,
                base_path=Path(self.data_path) / 'SeqCombMV'
            )

    def load_data(self, train_ratio=0.8):
        self.D = process_Synth(
            split_no = self.split_cv[0]+1,
            device = self.device, 
            base_path = Path(self.data_path) / 'SeqCombMV'
            )

        D = self.D

        # Extract training data
        X_train = self.D["train_loader"].X.permute(1, 2, 0) # N, D, T after this
        y_train = self.D["train_loader"].y

        # Extract test data (D['test'] is a tuple: (X, times, y))
        X_test = self.D["test"][0].permute(1, 2, 0)
        y_test = self.D["test"][2]

        # Get stats
        # ---- Helper: convert labels to integer class ids ----
        def to_class_ids(y: torch.Tensor) -> np.ndarray:
            """
            y: (N,) int labels or (N,C) one-hot/logits/probabilities
            returns: (N,) numpy int array of class ids
            """
            y = y.detach().cpu()
            if y.ndim == 2 and y.shape[1] > 1:
                # one-hot / probs / logits → class id
                y = y.argmax(dim=1)
            else:
                y = y.view(-1)
            return y.numpy().astype(int)

        y_train_np = to_class_ids(y_train)
        y_test_np  = to_class_ids(y_test)

        # ---- Count per-class, show counts and percentages ----
        all_classes = np.unique(np.concatenate([y_train_np, y_test_np]))

        def counts_for(arr: np.ndarray, classes: np.ndarray):
            vals, cnts = np.unique(arr, return_counts=True)
            d = {int(c): 0 for c in classes}
            for v, c in zip(vals, cnts):
                d[int(v)] = int(c)
            return d

        train_counts = counts_for(y_train_np, all_classes)
        test_counts  = counts_for(y_test_np,  all_classes)

        def with_pct(cnt_dict):
            total = sum(cnt_dict.values()) or 1
            return {k: {"n": v, "pct": round(100.0 * v / total, 2)} for k, v in cnt_dict.items()}

        print("Classes:", list(map(int, all_classes)))
        print("Train class stats:", with_pct(train_counts))
        print("Test  class stats:", with_pct(test_counts))


        self._get_loaders(X_train, y_train, X_test, y_test)


    def load_ground_truth_importance(self) -> np.ndarray:
        """
        Returns the ground-truth explanations from self.D['gt_exps'].
        Ensures self.D is loaded before access.
        """
        self.load_D()  # Make sure self.D is loaded

        if "gt_exps" not in self.D:
            raise ValueError("No 'gt_exps' found in dataset.")

        return np.asarray(self.D["gt_exps"].permute(1,2,0))

        # (200, 1000, 4)

    def get_name(self) -> str:
        return f"seqcombmv"

    @property
    def data_type(self) -> str:
        return "seqcombmv"


    @property
    def num_classes(self) -> int:
        return 4
    


################################
##### Real World Datasets  #####
################################

class Mimic(WinITDataset):
    """
    The pre-processed Mimic mortality dataset.
    Num Features = 31, Num Times = 48, Num Classes = 1
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name: str = "patient_vital_preprocessed.pkl",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name = file_name

    def load_data(self, train_ratio=0.8):
        with (self.data_path / self.file_name).open("rb") as f:
            data = pickle.load(f)
        feature_size = len(data[0][0])

        n_train = int(train_ratio * len(data))

        X = np.array([x for (x, y, z) in data])
        train_data = X[0:n_train]
        test_data = X[n_train:]
        train_label = np.array([y for (x, y, z) in data[0:n_train]])
        test_label = np.array([y for (x, y, z) in data[n_train:]])

        train_data, test_data = self.normalize(train_data, test_data, feature_size)

        self._get_loaders(train_data, train_label, test_data, test_label)

    @staticmethod
    def normalize(train_data, test_data, feature_size):
        d = np.stack([x.T for x in train_data], axis=0)
        num_timesteps = train_data.shape[-1]
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        train_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in train_data
            ]
        )
        test_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in test_data
            ]
        )
        return train_data, test_data

    def get_name(self) -> str:
        return "mimic"

    @property
    def data_type(self) -> str:
        return "mimic"

    @property
    def num_classes(self) -> int:
        return 1
    
class Boiler(WinITDataset):
    """
    The Boiler dataset
    Num Features = 20, Num Times = 36, Num Classes = 2, Samples: 90,115
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name: str = "patient_vital_preprocessed.pkl",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
        # split_no: ,
        device: str = 'cuda'
    ):
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name = file_name
        # self.split_no = split_no
        self.device = device
        self.data_path = data_path


    def load_data(self, train_ratio=0.8):
        D = process_Boiler_OLD(
                           split_no = '1',
                           device = self.device, 
                           train_ratio=train_ratio,
                        #    need_binarize = True, 
                        #    exclude_pac_pvc = True, 
                           base_path = Path(self.data_path) / 'Boiler')


        train, val, test = D

        # unpack your new dataset tuples
        # train[0] has shape (T, N_train, F)
        # train[2] has shape (N_train,)
        X_tr_raw, _, y_tr_raw = train
        X_va_raw, _, y_va_raw = val
        X_te_raw, _, y_te_raw = test

        # get dimensions
        T_tr, N_tr, F = X_tr_raw.shape
        T_te, N_te, _ = X_te_raw.shape

        assert F == X_te_raw.shape[2], "feature‐dim mismatch!"
        assert T_tr == T_te, "time‐dim must match across splits!"

        # permute into (N, F, T)
        X_train = X_tr_raw.permute(1, 2, 0)   # (N_train, F, T)
        X_test  = X_te_raw.permute(1, 2, 0)   # (N_test,  F, T)

        # expand labels to (N, T)
        y_train = y_tr_raw.unsqueeze(1).repeat(1, T_tr)  # (N_train, T)
        y_test  = y_te_raw.unsqueeze(1).repeat(1, T_tr)  # (N_test,  T)

        # now plug into your existing loader‐builder
        self._get_loaders(X_train, y_train, X_test, y_test)


        # self._get_loaders(X_train, y_train, X_test, y_test)

    @staticmethod
    def normalize(train_data, test_data, feature_size):
        d = np.stack([x.T for x in train_data], axis=0)
        num_timesteps = train_data.shape[-1]
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        train_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in train_data
            ]
        )
        test_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in test_data
            ]
        )
        return train_data, test_data

    def get_name(self) -> str:
        return "boiler"

    @property
    def data_type(self) -> str:
        return "boiler"

    @property
    def num_classes(self) -> int:
        return 1

class MITECG(WinITDataset):
    """
    The MITECG dataset
    Num Features = x, Num Times = 360, Num Classes = 1, Samples: ~90k
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name: str = "patient_vital_preprocessed.pkl",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
        # split_no: ,
        device: str = 'cuda'
    ):
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name = file_name
        # self.split_no = split_no
        self.device = device
        self.data_path = data_path


    def load_data(self, train_ratio=0.8): # TODO: need to fix the split number
        D = process_MITECG_for_WINIT(
                           device = self.device, 
                           hard_split = True, 
                           need_binarize = True, 
                           exclude_pac_pvc = True, 
                           base_path = Path(self.data_path) / 'MITECG')


        all_chunk = D

        # all_chunk.X shape: (features, n_samples, time_steps)? 
        # but your usage all_chunk.X[:, i] → shape (features, time_steps)
        n_time_steps, n_samples, n_features = all_chunk.X.shape

        # 1) Build lists of per‐sample tensors & labels
        X_list = []
        y_list = []
        for i in range(n_samples):
            # X[:, i, :] has shape (n_features, n_time_steps)
            x_i = all_chunk.X[:, i, :].to(torch.float32)  # ensure float32
            y_i = all_chunk.y[i].to(torch.int64)           # ensure int64
            X_list.append(x_i)                             # Tensor (features, time)
            y_list.append(y_i)                             # Tensor scalar                # → scalar Tensor

        # 2) Stack into big tensors
        X_tensor = torch.stack(X_list, dim=0).permute(0,2,1)  # shape (n_samples, n_features, n_time_steps)
        y_tensor = torch.stack(y_list, dim=0)   # shape (n_samples,)
        y_tensor = y_tensor.unsqueeze(1).repeat(1, n_time_steps)        # now (n_samples, n_time_steps)

        # 3) Train/test split
        y_flat = y_tensor[:, 0].cpu().numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor[:,0]

        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor,             # features
            y_tensor,             # full (n, T) label array
            train_size=train_ratio,
            stratify=y_flat,      # ensures both classes appear in both splits
            random_state=42,
        )

        # Now X_train, y_train are torch.Tensor (on original device)
        # To count classes, move labels to CPU+numpy first:
        y_train_np = y_train[:, 0].detach().cpu().numpy()
        y_test_np  = y_test[:,  0].detach().cpu().numpy()

        train_counts = np.unique(y_train_np, return_counts=True)
        test_counts  = np.unique(y_test_np,  return_counts=True)

        print("Train class counts:", dict(zip(train_counts[0], train_counts[1])))
        print("Test  class counts:", dict(zip(test_counts[0],  test_counts[1])))




        self._get_loaders(X_train, y_train, X_test, y_test)

    @staticmethod
    def normalize(train_data, test_data, feature_size):
        d = np.stack([x.T for x in train_data], axis=0)
        num_timesteps = train_data.shape[-1]
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        train_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in train_data
            ]
        )
        test_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in test_data
            ]
        )
        return train_data, test_data

    def get_name(self) -> str:
        return "mitecg"

    @property
    def data_type(self) -> str:
        return "mitecg"

    @property
    def num_classes(self) -> int:
        return 1
    


class PAM(WinITDataset): 
    """
    The PAM dataset
    Num Features = 17, Num Times = 600, Num Classes = 8, Samples: 5333
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name: str = "patient_vital_preprocessed.pkl",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
        # split_no: ,
        device: str = 'cuda'
    ):
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name = file_name
        # self.split_no = split_no
        self.device = device
        self.data_path = data_path
        self.split_cv = cv_to_use


    def load_data(self, train_ratio=0.8):
        train_chunk, val_chunk, test_chunk = process_PAM(
            split_no = self.split_cv[0]+1,
            device = self.device, 
            base_path = Path(self.data_path) / 'PAM',
            gethalf = False)


        # train_dataset = PAMDataset(train_chunk.X, train_chunk.time, train_chunk.y)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)


        X_train = train_chunk.X.permute(1,2,0)
        X_test = test_chunk.X.permute(1,2,0)
        X_valid = val_chunk.X.permute(1,2,0)

        y_train = train_chunk.y
        y_test = test_chunk.y
        y_valid = val_chunk.y

        self._get_loaders(X_train, y_train, X_test, y_test,  X_valid, y_valid)

    @staticmethod
    def normalize(train_data, test_data, feature_size):
        d = np.stack([x.T for x in train_data], axis=0)
        num_timesteps = train_data.shape[-1]
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        train_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in train_data
            ]
        )
        test_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in test_data
            ]
        )
        return train_data, test_data

    def get_name(self) -> str:
        return "pam"

    @property
    def data_type(self) -> str:
        return "pam"

    @property
    def num_classes(self) -> int:
        return 8





class Epilepsy(WinITDataset):
    """
    The Epilepsy dataset
    Num Features = 1, Num Times = 178, Num Classes = 2, Samples: 11500
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name: str = "patient_vital_preprocessed.pkl",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
        # split_no: ,
        device: str = 'cuda'
    ):
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name = file_name
        # self.split_no = split_no
        self.device = device
        self.data_path = data_path
        self.split_cv = cv_to_use


    def load_data(self, train_ratio=0.8):


        D = process_Epilepsy(split_no = self.split_cv[0]+1,
                         device = self.device,
                         base_path =  Path(self.data_path) / 'Epilepsy')


        train_chunk, val_chunk, test_chunk = D

        X_train = train_chunk.X.permute(1,2,0)
        X_test = test_chunk.X.permute(1,2,0)
        X_valid = val_chunk.X.permute(1,2,0)

        y_train = train_chunk.y
        y_test = test_chunk.y
        y_valid = val_chunk.y


        def print_class_stats(name, y):
            # convert to numpy if tensor
            y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else y
            classes, counts = np.unique(y_np, return_counts=True)
            total = len(y_np)
            print(f"\n{name} set class distribution:")
            for c, cnt in zip(classes, counts):
                print(f"  Class {c}: {cnt} samples ({cnt/total:.2%})")
            print(f"  Total: {total}")

        print_class_stats("Train", y_train)
        print_class_stats("Validation", y_valid)
        print_class_stats("Test", y_test)

        self._get_loaders(X_train, y_train, X_test, y_test,  X_valid, y_valid)

    @staticmethod
    def normalize(train_data, test_data, feature_size):
        d = np.stack([x.T for x in train_data], axis=0)
        num_timesteps = train_data.shape[-1]
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        train_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in train_data
            ]
        )
        test_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in test_data
            ]
        )
        return train_data, test_data

    def get_name(self) -> str:
        return "epilepsy"

    @property
    def data_type(self) -> str:
        return "epilepsy"

    @property
    def num_classes(self) -> int:
        return 1
    


