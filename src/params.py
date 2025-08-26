from __future__ import annotations


import itertools
import logging
import pathlib

from datetime import datetime
from typing import Dict, Any, List


import os

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import torch
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


from src.dataloader import Mimic, Boiler, MITECG, MITECG_Old, PAM, SimulatedSwitch, SimulatedState, SimulatedSpike, \
    WinITDataset, SimulatedData, SeqCombMV
from src.explainer.explainers import BaseExplainer, DeepLiftExplainer, IGExplainer, \
    GradientShapExplainer
from src.explainer.masker import Masker, Masker1




class Params:
    def __init__(self, argdict: Dict[str, Any]):
        self.argdict = argdict

        self._all_explainer_dict: Dict[str, List[Dict[str, Any]]] | None = None
        self._generators_to_train: Dict[str, List[Dict[str, Any]]] | None = None

        self._outpath: pathlib.Path | None = None
        self._ckptpath: pathlib.Path | None = None
        self._plotpath: pathlib.Path | None = None
        self._model_args: Dict[str, Any] | None = None
        self._model_train_args: Dict[str, Any] | None = None

        self._datasets = self._resolve_datasets()
        self._resolve_model_args()
        self._resolve_explainers()
        self._init_logging()

    @property
    def datasets(self) -> WinITDataset:
        return self._datasets

    @property
    def model_args(self) -> Dict[str, Any]:
        return {} if self._model_args is None else self._model_args

    @property
    def model_train_args(self) -> Dict[str, Any]:
        return {} if self._model_train_args is None else self._model_train_args

    @property
    def all_explainer_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        return {} if self._all_explainer_dict is None else self._all_explainer_dict

    @property
    def generators_to_train(self) -> Dict[str, List[Dict, str, Any]]:
        return {} if self._generators_to_train is None else self._generators_to_train

    @property
    def outpath(self) -> pathlib.Path | None:
        return None if self._outpath is None else self._outpath

    @property
    def ckptpath(self) -> pathlib.Path | None:
        return None if self._ckptpath is None else self._ckptpath

    @property
    def plotpath(self) -> pathlib.Path | None:
        return None if self._plotpath is None else self._plotpath

    def _resolve_datasets(self) -> WinITDataset:
        data = self.argdict["data"]
        testbs = self.argdict["testbs"]
        batch_size = self.argdict["batchsize"]
        data_path = self.argdict["datapath"]
        data_seed = self.argdict["dataseed"]
        cv_to_use = self.argdict["cv"]
        nondeterministic = self.argdict["nondeterministic"]
        kwargs = {"batch_size": batch_size, "seed": data_seed, "cv_to_use": cv_to_use,
                  "deterministic": not nondeterministic}


        if data_path is not None:
            kwargs["data_path"] = data_path

        if data == "spike":
            kwargs["testbs"] = 300 if testbs == -1 else testbs
            delay = self.argdict["delay"]
            return SimulatedSpike(delay=delay, **kwargs)

        if data == "mimic":
            kwargs["testbs"] = 1000 if testbs == -1 else testbs
            return Mimic(**kwargs)
        
        if (data == "boiler"):
            kwargs["testbs"] = 1000 if testbs == -1 else testbs
            return Boiler( **kwargs)
        
        if (data == "mitecg") or (data == "mitecg1"):
            kwargs["testbs"] = 1000 if testbs == -1 else testbs
            return MITECG( **kwargs)
        
        if (data == "mitecg_old") or (data == "mitecg1"):
            kwargs["testbs"] = 1000 if testbs == -1 else testbs
            return MITECG_Old( **kwargs)
        
        if (data == "pam"):
            kwargs["testbs"] = 1000 if testbs == -1 else testbs
            return PAM( **kwargs)

        if data == "switch":
            kwargs["testbs"] = 300 if testbs == -1 else testbs
            return SimulatedSwitch(**kwargs)

        if data == "state":
            kwargs["testbs"] = 300 if testbs == -1 else testbs
            return SimulatedState(**kwargs)
        
        
        if data == "seqcombmv":
            kwargs["testbs"] = 300 if testbs == -1 else testbs
            return SeqCombMV(**kwargs)

        raise ValueError(f"Unknown data {data}")

    def _resolve_explainers(self) -> None:
        explainers = self.argdict["explainer"]
        num_samples = self.argdict["num_samples"]
        windows = self.argdict["windows"]
        winit_metrics = self.argdict["winitmetric"]
        num_masks = self.argdict["num_masks"]

        all_explainer_dict = {}
        generator_dict = {}
        for explainer in explainers:
            if explainer == "dynamask":
                explainer_dict = self._resolve_dynamask_explainer_dict()
                all_explainer_dict[explainer] = [explainer_dict] 

            elif explainer in ["winit", "biwinit", "jimex"]:
                list_explainer_dict = []
                list_generator_dict = []

                for window in windows:
                    for num_mask in num_masks:
                        for winit_metric in winit_metrics:
                            


                            if (explainer == 'fit'):
                                single_explainer_dict = {
                                                            "window_size": window,
                                                            "joint": self.argdict["joint"],
                                                            "conditional": self.argdict["conditional"],
                                                            "usedatadist": self.argdict['usedatadist'],
                                                            "random_state": self.argdict["explainerseed"],
                                                            "metric": winit_metric
                                                        }
                                
                                list_explainer_dict.append(single_explainer_dict)


                            elif (explainer == 'winit'):

                                single_explainer_dict = {
                                                            "window_size": window,
                                                            "joint": self.argdict["joint"],
                                                            "conditional": self.argdict["conditional"],
                                                            "usedatadist": self.argdict['usedatadist'],
                                                            "random_state": self.argdict["explainerseed"],
                                                            "num_samples": self.argdict["num_samples"],
                                                            "metric": winit_metric

                                                        }
                                list_explainer_dict.append(single_explainer_dict)

                            elif (explainer == 'jimex'):

                                single_explainer_dict = {
                                                            "window_size": window,
                                                            "joint": self.argdict["joint"],
                                                            "conditional": self.argdict["conditional"],
                                                            "usedatadist": self.argdict['usedatadist'],
                                                            "random_state": self.argdict["explainerseed"],
                                                            "num_samples": self.argdict["num_samples"],
                                                            "metric": winit_metric,
                                                            "num_masks": num_mask,
                                                            "wt_bounds": self.argdict["wt_bounds"],
                                                            "wd_bounds": self.argdict["wd_bounds"],
                                                            "all_zero_cf": False if self.argdict["usedatadist"] else self.argdict["all_zero_cf"] 

                                                        }
                                list_explainer_dict.append(single_explainer_dict)

                    list_generator_dict.append(single_explainer_dict)

                # Deduplicate dicts

                def make_hashable(d):
                    """Convert dict values into hashable equivalents."""
                    return frozenset(
                        (k, tuple(v) if isinstance(v, list) else v) 
                        for k, v in d.items()
                    )
                list_explainer_dict = list({make_hashable(d): d for d in list_explainer_dict}.values())
                list_generator_dict = list({make_hashable(d): d for d in list_generator_dict}.values())

                all_explainer_dict[explainer] = list_explainer_dict
                generator_dict[explainer] = list_generator_dict

            else:
                explainer_dict = {}
                if explainer in ["fit", "fo", "afo"] and num_samples != -1:
                    explainer_dict["num_samples"] = num_samples
                if explainer == "fit":
                    generator_dict["fit"] = [explainer_dict]
                all_explainer_dict[explainer] = [explainer_dict]


        self._all_explainer_dict = all_explainer_dict
        self._generators_to_train = generator_dict

    def _resolve_dynamask_explainer_dict(self) -> Dict[str, Any]:
        data = self.argdict["data"]
        area = self.argdict["area"]
        loss = self.argdict["loss"]
        timereg = self.argdict["timereg"]
        sizereg = self.argdict["sizereg"]
        deletion_mode = self.argdict["deletion"]
        blur_type = self.argdict["blurtype"]
        use_last_timestep_only = self.argdict["lastonly"]
        explainer_dict = {"num_epoch": self.argdict["epoch"]}
        if loss is not None:
            explainer_dict["loss"] = loss
        if area is not None:
            explainer_dict["area_list"] = area
        elif data == "mimic":
            explainer_dict["area_list"] = [0.05]
        if timereg is not None:
            explainer_dict["time_reg_factor"] = timereg
        elif data == "mimic":
            explainer_dict["time_reg_factor"] = 0
        if sizereg is not None:
            explainer_dict["size_reg_factor_dilation"] = sizereg
        elif data == "mimic":
            explainer_dict["size_reg_factor_dilation"] = 10000
        if deletion_mode is not None:
            explainer_dict["deletion_mode"] = deletion_mode
        elif data == "mimic":
            explainer_dict["deletion_mode"] = True
        if blur_type is not None:
            explainer_dict["blur_type"] = blur_type
        elif data == "mimic":
            explainer_dict["blur_type"] = "fadema"
        if use_last_timestep_only is not None:
            explainer_dict["use_last_timestep_only"] = use_last_timestep_only == "True"
        elif data == "mimic":
            explainer_dict["use_last_timestep_only"] = False
        return explainer_dict

    def _resolve_model_args(self) -> None:
        ad = self.argdict
        get = lambda k, d=None: ad.get(k, d)

        # --- Normalize / validate model type ---
        model_type = str(get('modeltype', 'GRU')).upper()

        if model_type not in {'GRU', 'LSTM', 'CONV', 'TRANSFORMER'}:
            raise ValueError(f"Unsupported modeltype: {model_type}")

        # --- Collect relevant params per model from argdict (only keep if provided) ---
        keys_by_model = {
            'GRU': ['hidden_size', 'dropout', 'num_layers', 'bidirectional', 'input_size', 'n_classes'],
            'LSTM': ['hidden_size', 'dropout', 'num_layers', 'bidirectional', 'input_size', 'n_classes'],
            'CONV': ['in_channels', 'hidden_channels', 'kernel_size', 'stride', 'padding', 'dropout', 'n_classes'],
            'TRANSFORMER': ['d_inp', 'nhead', 'num_layers', 'trans_dim_feedforward', 'dropout',
                            'n_classes', 'd_pe']
        }




        model_args = {'model_type': model_type}
        for k in keys_by_model[model_type]:
            v = ad.get(k, None)
            if v is not None:
                model_args[k] = v

        # Backward-compat for your earlier keys
        # (only add if not already set)
        # if 'hidden_size' not in model_args and get('hiddensize') is not None:
        #     model_args['hidden_size'] = get('hiddensize')
        # if 'num_layers' not in model_args and get('numlayers') is not None:
        #     model_args['num_layers'] = get('numlayers')

        # --- Optim settings with sensible defaults ---
        lr = get('lr', None)


        weight_decay = float(get('weight_decay', 0.0))
        num_layers = ad.get('numlayers')
        # --- Epochs logic (preserve Mimic-specific defaults, else use epochs_classifier) ---
        epochs = get('epochs_classifier', None)


        # Finalize
        self._model_args = model_args
        self._model_train_args = {
            'num_epochs': int(epochs),
            'lr': float(lr),
            'weight_decay': weight_decay
        }


        base_out_path = pathlib.Path(self.argdict["outpath"])
        base_ckpt_path = pathlib.Path(self.argdict["ckptpath"])
        base_plot_path = pathlib.Path(self.argdict["plotpath"])
        self._outpath = self._resolve_path(base_out_path, model_type, num_layers)
        self._ckptpath = self._resolve_path(base_ckpt_path, model_type, num_layers)
        self._plotpath = self._resolve_path(base_plot_path, model_type, num_layers)

    def _init_logging(self) -> logging.Logger:
        format = '%(asctime)s %(levelname)8s %(name)25s: %(message)s'
        log_formatter = logging.Formatter(format)

        if self.argdict["logfile"] is None:
            time_str = datetime.now().strftime("%Y%m%d-%H%M")
            log_file_name = f"log_{time_str}.log"
        else:
            log_file_name = self.argdict["logfile"]

        log_path = pathlib.Path(self.argdict["logpath"])
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_name = log_path / log_file_name
        logging.basicConfig(format=format,
                            level=logging.getLevelName(self.argdict['loglevel'].upper()))

        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(str(log_file_name))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        return root_logger

    def _resolve_path(self, base_path: pathlib.Path, model_type: str, num_layers: int):
        if model_type == "GRU":
            return base_path / f"gru{num_layers}layer"
        elif model_type == "LSTM":
            return base_path / "lstm"
        elif model_type == "CONV":
            return base_path / "conv"
        elif model_type == "TRANSFORMER":
            return base_path / "transformer"
        else:
            raise Exception("Unknown model type ({})".format(model_type))

    def get_maskers(self, explainer: BaseExplainer) -> List[Masker]:
        maskers = []
        seed = self.argdict["maskseed"]
        absolutize = isinstance(explainer,
                                (DeepLiftExplainer, IGExplainer, GradientShapExplainer))
        for drop, aggregate_method in itertools.product(self.argdict["drop"],
                                                        self.argdict["aggregate"]):
            if drop == "bal":
                mask_methods = ["std"]
                top = self.argdict["top"]
                balanced = True
            elif drop == "local":
                mask_methods = self.argdict["mask"]
                top = self.argdict["top"]
                balanced = False
            else:
                mask_methods = self.argdict["mask"]
                top = self.argdict["toppc"]
                balanced = False

            for mask_method in mask_methods:
                maskers.append(
                    Masker1(mask_method, top, balanced, seed, absolutize, aggregate_method))
        return maskers
    

    def get_maskers1(self, 
                     explainer,
                     include_legacy = False) -> List[Masker1]:
        """
        Hybrid Masker factory — supports legacy `self.argdict`-driven config
        and new fixed maskers (cells/zero, cells/mean), plus optional legacy extras.

        Args:
            explainer: explainer instance (BaseExplainer or subclass)

        Returns:
            List[Masker]
        """
        maskers: List[Masker1] = []
        seed = self.argdict["maskseed"]

        # Legacy absolutize rule
        absolutize = isinstance(
            explainer,
            (DeepLiftExplainer, IGExplainer, GradientShapExplainer)
        )

        # === Legacy loop ===
        if (include_legacy):
            for drop, aggregate_method in itertools.product(self.argdict["drop"], self.argdict["aggregate"]):
                if drop == "bal":
                    mask_methods = ["std"]
                    top = self.argdict["top"]
                    balanced = True
                elif drop == "local":
                    mask_methods = self.argdict["mask"]
                    top = self.argdict["top"]
                    balanced = False
                else:
                    mask_methods = self.argdict["mask"]
                    top = self.argdict["toppc"]
                    balanced = False

                for mask_method in mask_methods:
                    maskers.append(
                        Masker1(mask_method, top, balanced, seed, absolutize, aggregate_method)
                    )

        # === New-style additions ===
        # Only add if not already present (avoid duplicates if argdict already covered them)
        new_maskers = [
            Masker1(
                mask_method="cells",

                balanced=False,
                seed=seed,
                absolutize=absolutize,
                aggregate_method="mean",
                substitution="zero"
            ),
            Masker1(
                mask_method="cells",

                balanced=False,
                seed=seed,
                absolutize=absolutize,
                aggregate_method="mean",
                substitution="mean"
            )
        ]
        for nm in new_maskers:
            if not any(
                (m.mask_method == nm.mask_method and getattr(m, "substitution", None) == getattr(nm, "substitution", None))
                for m in maskers
            ):
                maskers.append(nm)

        return maskers