from __future__ import annotations

import logging
import pathlib
import pickle as pkl
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataloader import WinITDataset, SimulatedData
from src.explainer.dynamaskexplainer import DynamaskExplainer
from src.explainer.masker import Masker
from src.explainer.masker import Masker1
from src.explainer.explainers import (
    BaseExplainer,
    IGExplainer,
    DeepLiftExplainer,
    FOExplainer,
    AFOExplainer,
    GradientShapExplainer,
    MockExplainer,
)
from src.explainer.fitexplainers import FITExplainer
from src.explainer.generator.generator import GeneratorTrainingResults
from src.explainer.original_winitexplainers import OGWinITExplainer
from src.explainer.biwinitexplainers import BiWinITExplainer
from src.explainer.jimex import JIMEx
from src.modeltrainer import ModelTrainerWithCv
from src.plot import BoxPlotter
from src.utils.basic_utils import aggregate_scores


class ExplanationRunner:
    """
    Our main class for training the model, training the generator, running explanations and
    evaluating explanations for various datasets.
    """

    def __init__(
        self,
        dataset: WinITDataset,
        device,
        out_path: pathlib.Path,
        ckpt_path: pathlib.Path,
        plot_path: pathlib.Path,
    ):
        """
        Constructor

        Args:
            dataset:
                The dataset wished to be run.
            device:
                The torch device.
            out_path:
                The path of the files containing the results of the explainer.
            ckpt_path:
                The path of the files containing the model and the generator checkpoints.
            plot_path:
                The path of the files containing plots or numpy arrays.
        """
        self.dataset = dataset
        self.device = device
        self.out_path = out_path
        self.ckpt_path = ckpt_path
        self.plot_path = plot_path
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        self.plot_path.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(ExplanationRunner.__name__)

        self.model_trainers: ModelTrainerWithCv | None = None
        self.explainers: Dict[int, BaseExplainer] | None = None
        self.importances: Dict[int, np.ndarray] | None = None

    def init_model(
        self,
        hidden_size: int = 200,
        dropout: float = 0.5,
        num_layers: int = 1,
        model_type: str = "GRU",
        nhead: int = None,
        trans_dim_feedforward: int = None,
        d_pe: str = None,
        # max_len: str = None,
        verbose_eval: int = 10,
        early_stopping: bool = False,
    ) -> None:
        """
        Initialize the base models.

        Args:
            hidden_size:
                The hidden size of the models
            dropout:
                The dropout rate of the models.
            num_layers:
                The number of layers of the models for GRU or LSTM.
            model_type:
                The model type of the models. GRU, LSTM or CONV.
            verbose_eval:
               Training metrics is logged at every given verbose_eval epoch.
            early_stopping:
               Whether apply early stopping or not.
        """
        self.model_trainers = ModelTrainerWithCv(
            self.dataset,
            self.ckpt_path,
            hidden_size,
            dropout,
            num_layers,
            nhead,
            trans_dim_feedforward,
            d_pe,
            # max_len,
            model_type,
            self.device,
            verbose_eval,
            early_stopping,
        )

    def train_model(
        self,
        num_epochs: int,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        use_all_times: bool = True,
    ) -> None:
        """
        Train the base models and log the test results.

        Args:
            num_epochs:
                The number of epochs to train the models.
            lr:
                The learning rate.
            weight_decay:
                The weight decay.
            use_all_times:
                Whether we use all timesteps or only the last timesteps to train the models.
        """
        if self.model_trainers is None:
            raise RuntimeError("Initialize the model first.")
        self.model_trainers.train_models(num_epochs, lr, weight_decay, use_all_times=use_all_times)
        self.model_trainers.load_model()
        self._get_test_results(use_all_times)

    def load_model(self, use_all_times: bool = True) -> None:
        """
        Load the base models and log the test results.

        Args:
            use_all_times:
                Whether we use all timesteps or only the last timesteps to train the models.
        """
        if self.model_trainers is None:
            raise RuntimeError("Initialize the model first.")
        self.model_trainers.load_model()
        self._get_test_results(use_all_times)

    def _get_test_results(self, use_all_times: bool) -> None:
        test_results = self.model_trainers.get_test_results(use_all_times)
        test_accs = [round(v.accuracy, 6) for v in test_results.values()]
        test_aucs = [round(v.auc, 6) for v in test_results.values()]
        test_auprcs = [round(v.AUPRC, 6) for v in test_results.values()]
        test_precisions = [round(v.precision, 6) for v in test_results.values()]
        test_recall = [round(v.recall, 6) for v in test_results.values()]
        self.log.info(f"Average Accuracy = {np.mean(test_accs):.4f}\u00b1{np.std(test_accs):.4f}")
        self.log.info(f"Average AUC = {np.mean(test_aucs):.4f}\u00b1{np.std(test_aucs):.4f}")
        self.log.info(f"Average AUPRC = {np.mean(test_auprcs):.4f}\u00b1{np.std(test_auprcs):.4f}")

        self.log.info(f"Model Accuracy on Tests = {test_accs}.")
        self.log.info(f"Model AUC on Tests = {test_aucs}.")
        self.log.info(f"Model Precision on Tests = {test_precisions}.")
        self.log.info(f"Model Recall on Tests = {test_recall}.")
        self.log.info(f"Model AUPRC on Tests = {test_auprcs}.")

    def run_inference(
        self,
        data: torch.Tensor | Dict[int, torch.Tensor] | None = None,
        with_activation: bool = True,
        return_all: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Run inference.

        Args:
            data:
                The data to be run. Shape of the batch = (num_samples, num_features, num_times).
                If it is a Tensor, it will be run inference on the data for all CVs. If it is a
                dictionary of CV to Tensor, it will run inference on the data with the model
                corresponding to the CV. If it is None, the inference will be run on the test set
                of the dataset.
            with_activation:
                Whether activation should be used.
            return_all:
                Whether return all the timesteps or the last one.

        Returns:
            A dictionary of CV to numpy arrays of shape (num_samples, num_classes, num_times)
            if return_all is True. Otherwise, the numpy arrays will be of shape
            (num_samples, num_classes).
        """
        return self.model_trainers.run_inference(data, with_activation, return_all)

    def clean_up(self, clean_importance=True, clean_explainer=True, clean_model=False):
        """
        Clean up.

        Args:
            clean_importance:
                indicate whether we clean the importance stored.
            clean_explainer:
                indicate whether we clean the explainer stored.
            clean_model:
                indicate whether we clean the model stored.
        """
        if clean_model and self.model_trainers is not None:
            del self.model_trainers
            self.model_trainers = None
        if clean_explainer and self.explainers is not None:
            del self.explainers
            self.explainers = None
        if clean_importance and self.importances is not None:
            del self.importances
            self.importances = None
        torch.cuda.empty_cache()

    def get_explainers(
        self,
        explainer_name: str,
        explainer_dict: Dict[str, Any],
    ) -> None:
        """
        A "routing" function to retrieve the corresponding explainer.

        Args:
            explainer_name:
                The name of the explainer.
            explainer_dict:
                The necessary args to initiate the explainer.
        """
        if explainer_name == "winit":
            train_loaders = (
                self.dataset.train_loaders if explainer_dict.get("usedatadist") is True else None
            )
            self.explainers = {}
            kwargs = explainer_dict.copy()
            if "usedatadist" in kwargs:
                kwargs.pop("usedatadist")
            for cv in self.dataset.cv_to_use():
                train_loader = train_loaders[cv] if train_loaders is not None else None
                self.explainers[cv] = OGWinITExplainer(
                    self.device,
                    self.dataset.feature_size,
                    self.dataset.get_name(),
                    path=self._get_generator_path(cv),
                    train_loader=train_loader,
                    **kwargs,
                )

        elif explainer_name == "biwinit":
            train_loaders = (
                self.dataset.train_loaders if explainer_dict.get("usedatadist") is True else None
            )
            self.explainers = {}
            kwargs = explainer_dict.copy()
            if "usedatadist" in kwargs:
                kwargs.pop("usedatadist")
            for cv in self.dataset.cv_to_use():
                train_loader = train_loaders[cv] if train_loaders is not None else None
                self.explainers[cv] = BiWinITExplainer(
                    self.device,
                    self.dataset.feature_size,
                    self.dataset.get_name(),
                    path=self._get_generator_path(cv),
                    train_loader=train_loader,
                    # height = kwargs['height'],
                    **kwargs,
                )

        elif explainer_name == "jimex":
            train_loaders = (
                self.dataset.train_loaders if explainer_dict.get("usedatadist") is True else None
            )
            self.explainers = {}
            kwargs = explainer_dict.copy()
            if "usedatadist" in kwargs:
                kwargs.pop("usedatadist")
            for cv in self.dataset.cv_to_use():
                train_loader = train_loaders[cv] if train_loaders is not None else None
                self.explainers[cv] = JIMEx(
                    self.dataset.feature_size,
                    num_samples=explainer_dict.get("num_samples"),
                    num_masks = explainer_dict.get("num_masks"),
                    wt_bounds = explainer_dict.get("wt_bounds"),
                    wd_bounds = explainer_dict.get("wd_bounds"),
                    device = self.device,
                    metric = explainer_dict.get("metric"),
                    window_size = explainer_dict.get("window_size"),
                    train_loader = train_loader,
                    random_state = explainer_dict.get("random_state"),
                    all_zero_cf = explainer_dict.get("all_zero_cf")


                )

        elif explainer_name == "fit":
            self.explainers = {}
            for cv in self.dataset.cv_to_use():
                self.explainers[cv] = FITExplainer(
                    self.device,
                    self.dataset.feature_size,
                    self.dataset.get_name(),
                    path=self._get_generator_path(cv),
                    **explainer_dict,
                )

        elif explainer_name == "ig":
            self.explainers = {
                cv: IGExplainer(self.device) for cv in self.dataset.cv_to_use()
            }

        elif explainer_name == "deeplift":
            self.explainers = {
                cv: DeepLiftExplainer(self.device) for cv in self.dataset.cv_to_use()
            }

        elif explainer_name == "fo":
            self.explainers = {
                cv: FOExplainer(self.device, **explainer_dict) for cv in self.dataset.cv_to_use()
            }

        elif explainer_name == "afo":
            self.explainers = {
                cv: AFOExplainer(self.device, self.dataset.train_loaders[cv], **explainer_dict)
                for cv in self.dataset.cv_to_use()
            }

        elif explainer_name == "gradientshap":
            self.explainers = {
                cv: GradientShapExplainer(self.device) for cv in self.dataset.cv_to_use()
            }

        elif explainer_name == "mock":
            self.explainers = {
                cv: MockExplainer() for cv in self.dataset.cv_to_use()
            }

        elif explainer_name == "dynamask":
            self.explainers = {
                cv: DynamaskExplainer(self.device, **explainer_dict) for cv in self.dataset.cv_to_use()
            }

        else:
            raise ValueError("%s explainer not defined!" % explainer_name)

    def train_generators(self, num_epochs: int) -> Dict[int, GeneratorTrainingResults] | None:
        """
        Train the generator if applicable. Test the generator and save the generator
        training results.

        Args:
            num_epochs:
                Train the generator for number of epochs.

        Returns:
            The generator training results. None if the explainer has no generator to train.
        """
        if self.explainers is None:
            raise RuntimeError("explainer is not initialized. Call get_explainer to initialize.")

        results = {}
        generator_array_path = self._get_generator_array_path()
        generator_array_path.mkdir(parents=True, exist_ok=True)
        for cv in self.dataset.cv_to_use():
            self.log.info(f"Training generator for cv={cv}")
            gen_result = self.explainers[cv].train_generators(
                self.dataset.train_loaders[cv], self.dataset.valid_loaders[cv], num_epochs
            )
            self.explainers[cv].test_generators(self.dataset.test_loader)
            if gen_result is not None:
                results[cv] = gen_result
                np.save(
                    generator_array_path / f"{gen_result.name}_train_loss_cv_{cv}.npy",
                    gen_result.train_loss_trends,
                )
                np.save(
                    generator_array_path / f"{gen_result.name}_valid_loss_cv_{cv}.npy",
                    gen_result.valid_loss_trends,
                )
                np.save(
                    generator_array_path / f"{gen_result.name}_best_epoch_cv_{cv}.npy",
                    gen_result.best_epochs,
                )
        if len(results) > 0:
            return results
        return None

    def load_generators(self):
        """
        Load the generator and print the test results.
        """
        if self.explainers is None:
            raise RuntimeError("explainer is not initialized. Call get_explainer to initialize.")

        for cv in self.dataset.cv_to_use():
            self.explainers[cv].load_generators()
            self.explainers[cv].test_generators(self.dataset.test_loader)

    def _get_generator_path(self, cv: int) -> pathlib.Path:
        return self.ckpt_path / self.dataset.get_name() / str(cv)

    def _get_importance_path(self) -> pathlib.Path:
        return self.out_path / self.dataset.get_name()

    def _get_importance_file_name(self, cv: int):
        return f"{self.explainers[cv].get_name()}_test_importance_scores_{cv}.pkl"

    def set_model_for_explainer(self, set_eval: bool = True):
        """
        Set the base model for the explainer.

        Args:
            set_eval:
                Set the model to eval mode. If False, leave the model as is.
        """
        if self.explainers is None:
            raise RuntimeError("explainer is not initialized. Call get_explainer to initialize.")

        for cv in self.dataset.cv_to_use():
            self.explainers[cv].set_model(
                self.model_trainers.model_trainers[cv].model, set_eval=set_eval
            )

    def run_attributes(self) -> None:
        """
        Run attribution method for the explainer on the test set.
        """
        if self.explainers is None:
            raise RuntimeError("explainer is not initialized. Call get_explainer to initialize.")

        importances, extra_scores = self._run_attributes_recursive(self.dataset.test_loader)
        self.importances = importances
        self.extra_scores = extra_scores

    def _run_attributes_recursive(self, dataloader: DataLoader) -> Dict[int, np.ndarray]:
        """
        A convenient method to run attributes when we adjust the batch size if cuda is out of
        memory

        Args:
            dataloader:
                The data loader for the input for attributes

        Returns:
            A dictionary of CV to the attribution.
        """
        try:
            return self._run_attributes(dataloader)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # reduce batch size
                new_batch_size = dataloader.batch_size // 2
                if 0 < new_batch_size < dataloader.batch_size:
                    self.log.warning(
                        f"CUDA out of memory! Reducing batch size from "
                        f"{dataloader.batch_size} to {new_batch_size}"
                    )
                    new_loader = DataLoader(dataloader.dataset, new_batch_size)
                    # self.test_loader.batch_size = new_batch_size
                    return self._run_attributes_recursive(new_loader)
            raise e

    def _run_attributes(self, dataloader: DataLoader) -> Dict[int, np.ndarray]:
        """
        Run feature attribution.

        Args:
            dataloader:
                The data loader for the input for attributes

        Returns:
            A dictionary of CV to the attribution.
        """
        all_importance_scores = {}
        all_iSab = {}  # list of dicts, one per extra output
        all_IS = {}  # list of dicts, one per extra output

        for cv in self.dataset.cv_to_use():
            importance_list = []
            iSab_list     = [] 
            IS_list     = []  

            for x, y in tqdm(dataloader, desc=f"CV={cv} batches", total=len(dataloader)):
                x = x.to(self.device)

                # unpack however many things .attribute() returns
                ret = self.explainers[cv].attribute(x)
                if isinstance(ret, (tuple, list)):
                    # unpack exactly three if available, otherwise pad with Nones
                    score = ret[0]
                    extra1 = ret[1] if len(ret) > 1 else None
                    extra2 = ret[2] if len(ret) > 2 else None
                    iSab_list.append(extra1)
                    IS_list.append(extra2)
                else:
                    # ret is just a single array
                    score = ret
                    extra1 = None
                    extra2 = None

                importance_list.append(score)
                
            # concatenate along batch/time dimension
            importance_scores = np.concatenate(importance_list, axis=0)
            if iSab_list:
                iSab_scores = np.concatenate(iSab_list, axis=0)
            else:
                iSab_scores = np.empty((0,))
            if IS_list:
                IS_scores = np.concatenate(IS_list, axis=0)
            else:
                IS_scores = np.empty((0,))
            
            all_importance_scores[cv] = importance_scores
            all_iSab[cv] = iSab_scores
            all_IS[cv] = IS_scores

            
        return all_importance_scores, {'iSab': all_iSab, 'IS': all_IS}
       
    def save_importance(self):
        """
        Save the feature importance.
        """
        if self.importances is None:
            return
        importance_path = self._get_importance_path()
        importance_path.mkdir(parents=True, exist_ok=True)

        for cv, importance_scores in self.importances.items():
            importance_file_name = importance_path / self._get_importance_file_name(cv)
            self.log.info(f"Saving file to {importance_file_name}")
            with importance_file_name.open("wb") as f:
                pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)
            
        # do the same for extra_scores
        if getattr(self, "extra_scores", None) is not None:
            # if its a list
            if not isinstance(self.extra_scores, dict):
                fname = f"extra_scores.pkl"
                file_path = importance_path / fname
                self.log.info(f"Saving extra for CV={cv} to {file_path}")
                with file_path.open("wb") as f:
                    pkl.dump(self.extra_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

            # if a dict
            else:
                for cv, extras in self.extra_scores.items():
                    # if extras is a dict of named arrays
                    if isinstance(extras, dict):
                        for name, arr in extras.items():
                            fname = f"{cv}_{name}.pkl"
                            file_path = importance_path / fname
                            self.log.info(f"Saving extra '{name}' for CV={cv} to {file_path}")
                            with file_path.open("wb") as f:
                                pkl.dump(arr, f, protocol=pkl.HIGHEST_PROTOCOL)

                    # if extras is a list of arrays, just index them
                    elif isinstance(extras, (list, tuple)):
                        for idx, arr in enumerate(extras):
                            fname = f"{cv}_extra_{idx}.pkl"
                            file_path = importance_path / fname
                            self.log.info(f"Saving extra index {idx} for CV={cv} to {file_path}")
                            with file_path.open("wb") as f:
                                pkl.dump(arr, f, protocol=pkl.HIGHEST_PROTOCOL)

                    else:
                        # fallback: try saving it directly
                        fname = f"{cv}_extra_scores.pkl"
                        file_path = importance_path / fname
                        self.log.info(f"Saving extra_scores for CV={cv} to {file_path}")
                        with file_path.open("wb") as f:
                            pkl.dump(extras, f, protocol=pkl.HIGHEST_PROTOCOL)

    def load_importance(self):
        """
        Load the importance from the file.
        """
        importances = {}
        for cv in self.dataset.cv_to_use():
            importance_file_name = self._get_importance_path() / self._get_importance_file_name(cv)
            with importance_file_name.open("rb") as f:
                importance_scores = pkl.load(f)
            importances[cv] = importance_scores
        self.importances = importances

    def evaluate_simulated_importance(self, aggregate_methods) -> pd.DataFrame:
        """
        Run evaluation for importance for Simulated Data. The metrics are AUROC, AVPR, AUPRC,
        mean rank, mean rank (min) and positive ratios. Save the example boxes as well.

        Args:
            aggregate_methods:
                The aggregation method for WinIT.

        Returns:
            A DataFrame of shape (num_cv * aggregate_method, num_metric=6).
        """
        if not isinstance(self.dataset, SimulatedData):
            raise ValueError("non simulated dataset does not have simulated importances.")

        if self.importances is None:
            raise ValueError(
                "No importances is loaded. Call load_importance or run_attribute first."
            )

        ground_truth_importance = self.dataset.load_ground_truth_importance()

        absolutize = isinstance(
            next(iter(self.explainers.values())),
            (DeepLiftExplainer, IGExplainer, GradientShapExplainer),
        )
        df = self._evaluate_importance_with_gt(
            ground_truth_importance, absolutize, aggregate_methods
        )
        self._plot_boxes(num_to_plot=20, aggregate_methods=aggregate_methods)
        return df

    def evaluate_performance_drop(
        self,
        maskers: List[Masker],
        use_last_time_only=True,
    ) -> pd.DataFrame:
        """
        Evaluate the importances on non simulated dataset by performance drop. The metrics are
        AUC Drop, Pred Diff and avg Masked count.

        Args:
            maskers:
               The list of masking methods (maskers) that we are evaluating.
            use_last_time_only:
               The performance drop is only to use the last timestep only.

        Returns:
            A DataFrame object of shape (num_cv * num_maskers, num_metrics=3)
        """
        testset = list(self.dataset.test_loader.dataset)
        orig_preds = self.run_inference(self.dataset.test_loader, return_all=False)
        x_test = torch.stack(([x[0] for x_ind, x in enumerate(testset)])).cpu().numpy()
        y_test = torch.stack(([x[1] for x_ind, x in enumerate(testset)])).cpu().numpy()
        dfs = {}
        for masker in maskers:
            self.log.info(f"Beginning performance drop for mask={masker.get_name()}")
            new_xs = masker.mask(x_test, self.importances)
            new_xs = {k: torch.from_numpy(v) for k, v in new_xs.items()}
            self._plot_boxes(
                num_to_plot=20, aggregate_methods=[masker.aggregate_method], x_other=new_xs, mask_name=masker.get_name()
            )
            new_preds = self.run_inference(new_xs, return_all=False)
            df = pd.DataFrame()
            for cv in self.dataset.cv_to_use():
                orig_pred = orig_preds[cv]
                new_pred = new_preds[cv]
                if use_last_time_only:
                    orig_pred = orig_pred[:, -1]
                    new_pred = new_pred[:, -1]
                    if y_test.ndim == 2:
                        y_test = y_test[:, -1]
                orig_pred = orig_pred.reshape(-1)
                new_pred = new_pred.reshape(-1)
                y_test = y_test.reshape(-1)

                original_auc = metrics.roc_auc_score(y_test, orig_pred, average="macro")
                modified_auc = metrics.roc_auc_score(y_test, new_pred, average="macro")
                self.log.info(
                    f"cv={cv}, original auc={original_auc:.8f}, modified auc={modified_auc:.8f}"
                )

                avg_pred_diff = np.abs(orig_pred - new_pred).mean().item()
                auc_drop = original_auc - modified_auc
                avg_mask_count = (masker.all_masked_count[cv].sum() / len(x_test)).item()

                mask_array_path = self._get_mask_array_path()
                mask_array_path.mkdir(parents=True, exist_ok=True)
                array_prefix = f"{self.get_explainer_name()}_{masker.get_name()}_"
                np.save(
                    mask_array_path / f"{array_prefix}start_mask_cv_{cv}",
                    masker.start_masked_count[cv].sum(axis=0),
                )
                np.save(
                    mask_array_path / f"{array_prefix}all_mask_cv_{cv}",
                    masker.all_masked_count[cv].sum(axis=0),
                )

                df[cv] = pd.Series(
                    {
                        "auc_drop": auc_drop,
                        "avg_pred_diff": avg_pred_diff,
                        "avg_masked_count": avg_mask_count,
                    }
                )
            df = df.transpose()
            df.index.name = "cv"
            dfs[masker.get_name()] = df
        dfs = pd.concat(dfs, axis=0)
        dfs.index.name = "mask method"
        return dfs

    def _plot_boxes(
        self,
        num_to_plot,
        aggregate_methods: List[str],
        x_other: Dict[int, torch.Tensor] | None = None,
        mask_name: str = "",
    ) -> None:
        """
        Convenient method to plot and save the boxes for the importances, labels, data and
        predictions.

        Args:
            num_to_plot:
                The number of samples to plot.
            aggregate_methods:
                The aggregation method of importances for WinIT.
            x_other:
                Other data to plot. In case of Mimic masking, this gives the option to plot the
                masked data and the masked predictions.
            mask_name:
                The name of the mask.
        """
        explainer_name = self.get_explainer_name()
        plotter = BoxPlotter(self.dataset, self.plot_path, num_to_plot, explainer_name)
        if self.importances is not None:
            for aggregate_method in aggregate_methods:
                plotter.plot_importances(self.importances, aggregate_method)

        plotter.plot_labels()
        inference = self.run_inference()
        plotter.plot_x_pred(x=None, preds=inference)

        if isinstance(self.dataset, SimulatedData):
            ground_truth_importance = self.dataset.load_ground_truth_importance()
            plotter.plot_ground_truth_importances(ground_truth_importance)

        if x_other is not None:
            inference_other = self.run_inference(x_other)
            prefix = f"{explainer_name}_{mask_name}_masked"
            plotter.plot_x_pred(x_other, inference_other, prefix=prefix)

    def _evaluate_importance_with_gt(
        self, ground_truth_importance: np.ndarray, absolutize: bool, aggregate_methods: List[str]
    ):
        print("ground_truth_importance.shape:", ground_truth_importance.shape)
        ground_truth_importance = ground_truth_importance[:, :, 1:].reshape(
            len(ground_truth_importance), -1
        )
        print("ground_truth_importance.shape:", ground_truth_importance.shape)

        dfs = {}
        for aggregate_method in aggregate_methods:
            df = pd.DataFrame()
            for cv, importance_unaggregated in self.importances.items():

                importance_scores = aggregate_scores(importance_unaggregated, aggregate_method)
                print("importance_scores.shape:", importance_scores.shape)
                importance_scores = importance_scores[:, :, 1:].reshape(len(importance_scores), -1)
                print("importance_scores.shape:", importance_scores.shape)
                
                


                # compute mean ranks
                ranks = rankdata(-importance_scores, axis=1)
                ranks_min = rankdata(-importance_scores, axis=1, method="min")
                gt_positions = np.where(ground_truth_importance)
                gt_ranks = ranks[gt_positions]
                gt_ranks_min = ranks_min[gt_positions]
                mean_rank = np.mean(gt_ranks)
                mean_rank_min = np.mean(gt_ranks_min)

                gt_score = ground_truth_importance.flatten()
                print("gt_score.shape:", gt_score.shape)

                explainer_score = importance_scores.flatten()
                print("explainer_score.shape:", explainer_score.shape)

                if absolutize:
                    explainer_score = np.abs(explainer_score)

                if np.any(np.isnan(explainer_score)):
                    self.log.warning("NaNs appear in explainer scores!")
                explainer_score = np.nan_to_num(explainer_score)


                binary_case = (gt_score.ndim == 1)

                if binary_case:
                    auc_score = metrics.roc_auc_score(gt_score, explainer_score)
                    AP_score = metrics.average_precision_score(gt_score, explainer_score)
                    prec_score, rec_score, thresholds = metrics.precision_recall_curve(
                        gt_score, explainer_score
                    )
                    auprc_score = metrics.auc(rec_score, prec_score) if rec_score.shape[0] > 1 else -1

                    pos_ratio = ground_truth_importance.sum() / len(ground_truth_importance)

                    # ---------- AUP & AUR (integrate over fraction selected) ----------
                    # Sort by explainer score descending; sweep a threshold to include top-k features
                    order = np.argsort(-explainer_score, kind="mergesort")
                    gt_sorted = gt_score[order]  # 1 if truly important, else 0

                    n = gt_sorted.size
                    P = gt_sorted.sum()  # number of truly important features

                    if n > 0:
                        k = np.arange(1, n + 1)                # prefix lengths
                        x = k / n                               # fraction selected (0→1)
                        cum_tp = np.cumsum(gt_sorted)

                        # precision@k and recall@k for each prefix
                        precision_k = cum_tp / k
                        recall_k = (cum_tp / P) if P > 0 else np.zeros_like(cum_tp, dtype=float)

                        # numeric integration over x (fraction selected)
                        # trapezoidal rule; gives scalar areas in [0,1]
                        AUP = float(np.trapz(precision_k, x))
                        AUR = float(np.trapz(recall_k, x))
                    else:
                        AUP, AUR = 0.0, 0.0
                

                else:
                        # ---------------------- MULTICLASS (macro) ----------------------
                        # Expect shapes (N, C)
                        if explainer_score.ndim == 1:
                            raise ValueError("Multiclass detected (gt_score is 2D) but explainer_score is 1D.")
                        if gt_score.shape != explainer_score.shape:
                            raise ValueError(f"gt_score shape {gt_score.shape} != explainer_score shape {explainer_score.shape}")

                        # AUROC / AUPRC macro across classes
                        auc_score = metrics.roc_auc_score(gt_score, explainer_score, average='macro', multi_class='ovr')
                        AP_score = metrics.average_precision_score(gt_score, explainer_score, average='macro')

                        # AUPRC curve needs per-class handling; macro average areas
                        auprc_vals = []
                        for c in range(gt_score.shape[1]):
                            prec_c, rec_c, _ = metrics.precision_recall_curve(gt_score[:, c], explainer_score[:, c])
                            auprc_vals.append(metrics.auc(rec_c, prec_c) if rec_c.shape[0] > 1 else -1)
                        auprc_score = float(np.mean(auprc_vals)) if len(auprc_vals) else -1.0

                        # Pos ratio macro (mean positives per class)
                        pos_ratio = float(np.mean(gt_score.mean(axis=0)))

                        # AUP & AUR: compute per class using your same integration, then macro average
                        AUP_list, AUR_list = [], []
                        for c in range(gt_score.shape[1]):
                            gs = gt_score[:, c].astype(float)
                            es = explainer_score[:, c].astype(float)

                            order = np.argsort(-es, kind="mergesort")
                            gs_sorted = gs[order]

                            n = gs_sorted.size
                            P = gs_sorted.sum()

                            if n > 0:
                                k = np.arange(1, n + 1)
                                x = k / n
                                cum_tp = np.cumsum(gs_sorted)
                                precision_k = cum_tp / k
                                recall_k = (cum_tp / P) if P > 0 else np.zeros_like(cum_tp, dtype=float)
                                AUP_list.append(float(np.trapz(precision_k, x)))
                                AUR_list.append(float(np.trapz(recall_k, x)))
                            else:
                                AUP_list.append(0.0)
                                AUR_list.append(0.0)

                        AUP = float(np.mean(AUP_list))
                        AUR = float(np.mean(AUR_list))
                
                
                
                
                result = {
                    "AUROC": auc_score,
                    "AP": AP_score,
                    "AUPRC": auprc_score,
                    "AUP": AUP,            # <-- added
                    "AUR": AUR,            # <-- added
                    "Mean rank": mean_rank,
                    "Mean rank (min)": mean_rank_min,
                    "Pos ratio": pos_ratio,
                }
                self.log.info(f"cv={cv}")
                for k, v in result.items():
                    self.log.info(f"{k:20}: {v:.4f}")
                df[cv] = pd.Series(result)
            df = df.transpose()
            df.index.name = "cv"
            dfs[aggregate_method] = df
        df_all = pd.concat(dfs, axis=0)
        df_all.index.name = "aggregate method"
        return df_all


    def evaluate_performance_drop1(
        self,
        maskers: List[Masker],
        use_last_time_only: bool = True,
        k_values: list[float] = (0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99),
        directions: tuple[str, ...] = ("top", "bottom"),
    ) -> pd.DataFrame:
        """
        Real-world evaluation with no GT:
        For each masker, direction ('top'/'bottom') and k in k_values,
        - build masked inputs via Masker.mask(..., k, direction, mode='remove'|'keep')
        - compute metrics per CV:
            auc_drop, avg_pred_diff, avg_masked_count, comp_k, suff_k
        Returns:
        MultiIndexed DataFrame: (mask method, cv, direction, k) -> metrics
        """

        # ---------------- helpers ----------------
        def _slice_last(arr):
            return arr[:, -1] if (use_last_time_only and arr.ndim == 2) else arr

        def _bin_prob(p):
            # If multiclass (N,C) → use column 1 for ROC-AUC; else flatten
            return p[:, 1] if (p.ndim == 2 and p.shape[1] > 1) else p.reshape(-1)

        def _prob_of_predclass(orig_probs: np.ndarray, new_probs: np.ndarray):
            """
            Return per-sample probability of the ORIGINAL predicted class
            for both original and new probs.
            Handles binary (N,) or (N,1) and multiclass (N,C).
            """
            if orig_probs.ndim == 1:
                yhat = (orig_probs >= 0.5).astype(np.int64)
                p0 = np.where(yhat == 1, orig_probs, 1.0 - orig_probs)
                pn = np.where(yhat == 1, new_probs.reshape(-1), 1.0 - new_probs.reshape(-1))
                return p0, pn
            if orig_probs.ndim == 2 and orig_probs.shape[1] == 1:
                o = orig_probs[:, 0]
                n = new_probs.reshape(-1)
                yhat = (o >= 0.5).astype(np.int64)
                p0 = np.where(yhat == 1, o, 1.0 - o)
                pn = np.where(yhat == 1, n, 1.0 - n)
                return p0, pn
            # multiclass
            yhat = np.argmax(orig_probs, axis=1)
            idx = np.arange(orig_probs.shape[0])
            return orig_probs[idx, yhat], new_probs[idx, yhat]

        # ---------------- data & original preds ----------------
        testset = list(self.dataset.test_loader.dataset)
        x_test = torch.stack([x[0] for x in testset]).cpu().numpy()  # (N,F,T)
        y_test = torch.stack([x[1] for x in testset]).cpu().numpy()  # (N,) or (N,C)
        orig_preds = self.run_inference(self.dataset.test_loader, return_all=False)

        results_by_masker = {}

        for masker in maskers:
            self.log.info(f"[PerfDrop] Masker={masker.get_name()} | directions={directions} | k={k_values}")
            rows = []

            for direction in directions:
                for k in k_values:
                    # ---- REMOVE (for auc_drop, avg_pred_diff, comp_k) ----
                    new_xs_remove = masker.mask(
                        x_test, self.importances, k=k, direction=direction, mode="remove"
                    )
                    new_xs_remove = {cv: torch.from_numpy(v) for cv, v in new_xs_remove.items()}
                    new_preds_remove = self.run_inference(new_xs_remove, return_all=False)

                    # ---- KEEP-ONLY (for suff_k) ----
                    new_xs_keep = masker.mask(
                        x_test, self.importances, k=k, direction=direction, mode="keep"
                    )
                    new_xs_keep = {cv: torch.from_numpy(v) for cv, v in new_xs_keep.items()}
                    new_preds_keep = self.run_inference(new_xs_keep, return_all=False)

                    # ---- per-CV metrics ----
                    df = pd.DataFrame()
                    for cv in self.dataset.cv_to_use():
                        orig_pred = _slice_last(orig_preds[cv])
                        rem_pred  = _slice_last(new_preds_remove[cv])
                        keep_pred = _slice_last(new_preds_keep[cv])
                        y_vec     = _slice_last(y_test).reshape(-1)

                        # Ensure numpy for metrics
                        o_np = orig_pred if isinstance(orig_pred, np.ndarray) else orig_pred.detach().cpu().numpy()
                        r_np = rem_pred  if isinstance(rem_pred,  np.ndarray) else rem_pred.detach().cpu().numpy()
                        k_np = keep_pred if isinstance(keep_pred, np.ndarray) else keep_pred.detach().cpu().numpy()
                        y_np = y_vec if isinstance(y_vec, np.ndarray) else y_vec

                        # AUCs (macro if shapes permit; else binary column)
                        try:
                            auc_orig = metrics.roc_auc_score(y_np, o_np, average="macro")
                            auc_rem  = metrics.roc_auc_score(y_np, r_np, average="macro")
                        except Exception:
                            auc_orig = metrics.roc_auc_score(y_np, _bin_prob(o_np))
                            auc_rem  = metrics.roc_auc_score(y_np, _bin_prob(r_np))

                        auc_drop = float(auc_orig - auc_rem)
                        avg_pred_diff = float(np.mean(np.abs(_bin_prob(o_np) - _bin_prob(r_np))))

                        # Comprehensiveness: avg drop in predicted-class prob after REMOVE
                        p0, pR = _prob_of_predclass(o_np, r_np)
                        comp_k = float(np.mean(p0 - pR))

                        # Sufficiency: avg drop when KEEP-ONLY
                        p0, pK = _prob_of_predclass(o_np, k_np)
                        suff_k = float(np.mean(p0 - pK))

                        # Avg masked count from masker’s accounting (for REMOVE path)
                        try:
                            avg_masked = float((masker.all_masked_count[cv].sum() / len(x_test)).item())
                        except Exception:
                            avg_masked = np.nan

                        df.loc[cv, ["auc_drop", "avg_pred_diff", "avg_masked_count", "comp_k", "suff_k"]] = [
                            auc_drop, avg_pred_diff, avg_masked, comp_k, suff_k
                        ]

                        # (optional) persist masks for later analysis
                        mask_dir = self._get_mask_array_path()
                        mask_dir.mkdir(parents=True, exist_ok=True)
                        prefix = f"{self.get_explainer_name()}_{masker.get_name()}_{direction}_k={k}_"
                        try:
                            np.save(mask_dir / f"{prefix}start_mask_cv_{cv}", masker.start_masked_count[cv].sum(axis=0))
                            np.save(mask_dir / f"{prefix}all_mask_cv_{cv}",   masker.all_masked_count[cv].sum(axis=0))
                        except Exception:
                            pass

                    df.index.name = "cv"
                    df["direction"] = direction
                    df["k"] = k
                    df["masker_name"] = masker.get_name()
                    rows.append(df)

            out = pd.concat(rows, axis=0)
            results_by_masker[masker.get_name()] = out.set_index(["direction", "k"], append=True)

        final = pd.concat(results_by_masker, axis=0)
        final.index.names = ["mask method", "cv", "direction", "k"]
        return final



    def _get_mask_array_path(self) -> pathlib.Path:
        return self.plot_path / self.dataset.get_name() / "array"

    def _get_generator_array_path(self) -> pathlib.Path:
        return self.plot_path / self.dataset.get_name() / "generator_array"

    def get_explainer_name(self) -> str:
        if self.explainers is None:
            return ""
        return next(iter(self.explainers.values())).get_name()
