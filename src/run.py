from __future__ import annotations

import argparse

import logging

import time
from datetime import datetime


import pandas as pd
from omegaconf import OmegaConf
import os
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import torch.cuda

from src.dataloader import Mimic, Boiler, MITECG, PAM, SimulatedSwitch, SimulatedState, SimulatedSpike, \
    WinITDataset, SimulatedData, SeqCombMV
# from src.explainer.explainers import BaseExplainer, DeepLiftExplainer, IGExplainer, \
#     GradientShapExplainer
# from src.explainer.masker import Masker, Masker1
from src.explanationrunner import ExplanationRunner
from src.utils.basic_utils import append_df_to_csv
# from src.utils.get_masks import get_maskers, get_maskers1
from src.datagen.spikes_data_new import SpikeTrainDataset 
from src.params import Params




if __name__ == '__main__':

    # 1) parse only the --config argument (leave all other CLI args alone)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", "-c", type=str, default="config/main.yaml", help="Path to the OmegaConf YAML file",)
    args, remaining_argv = parser.parse_known_args()

    # 2) load the chosen YAML
    cfg = OmegaConf.load(args.config)

    argdict = OmegaConf.to_container(cfg, resolve=True)

    # argdict = vars(parser.parse_args())
    data = argdict['data']
    explainers = argdict['explainer']
    do_explain = argdict['do_explain']
    eval_explain = argdict['eval_explain']
    train_models = argdict['train']
    train_gen = argdict['traingen']
    result_file = argdict["resultfile"]
    epoch_gen = argdict["epoch_gen"]
    train_ratio = argdict.get("train_ratio") or 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # parse the arg
    params = Params(argdict)
    dataset = params.datasets
    model_args = params.model_args
    model_train_args = params.model_train_args
    all_explainer_dict = params.all_explainer_dict
    out_path = params.outpath
    ckpt_path = params.ckptpath
    plot_path = params.plotpath
    start_time = time.time()
    log = logging.getLogger("Base")
    for k, v in argdict.items():
        log.info(f"{k:15}: {v}")
    first = True
    save_failed = False

    all_df = []
    try:
        # load data and train model
        dataset.load_data(train_ratio=train_ratio)
        runner = ExplanationRunner(dataset, device, out_path, ckpt_path, plot_path)

        print(model_args.keys())
        runner.init_model(**model_args)
        use_all_times = not isinstance(dataset, (SeqCombMV, Mimic, Boiler, MITECG, PAM))
        if train_models:
            runner.train_model(**model_train_args, use_all_times=use_all_times)
        else:
            runner.load_model(use_all_times)
        log.info("Load data and train/load model done.")

        # train generators
        if train_gen:
            print('training gen for {} epochs'.format(epoch_gen))
            generators_to_train = params.generators_to_train
            for explainer_name, explainer_dict_list in generators_to_train.items():
                for explainer_dict in explainer_dict_list:
                    runner.get_explainers(explainer_name, explainer_dict=explainer_dict)
                    log.info(
                        f"Training Generator...Data={dataset.get_name()}, Explainer={explainer_name}")
                    runner.train_generators(num_epochs=epoch_gen)
            log.info("Training Generator Done.")

        for explainer_name, explainer_list in all_explainer_dict.items():
            for explainer_dict in explainer_list:
                # generate feature importance
                runner.clean_up(clean_importance=True, clean_explainer=True, clean_model=False)
                runner.get_explainers(explainer_name, explainer_dict=explainer_dict)
                runner.set_model_for_explainer(set_eval=explainer_name != "fit")

                if do_explain:
                    log.info(f"Running Explanations..."
                             f"Data={dataset.get_name()}, Explainer={explainer_name}, Dict={explainer_dict}")

                    # runner.load_generators()
                    runner.run_attributes()
                    runner.save_importance()
                    importances = runner.importances
                    log.info("Explanations done.")

                # evaluate importances
                if eval_explain:
                    log.info("Evaluating importance...")
                    log.info(f"Data={dataset.get_name()}, Explainer={explainer_name}")
                    if runner.importances is None:
                        runner.load_importance()
                    if isinstance(dataset, SimulatedData):
                        df = runner.evaluate_simulated_importance(argdict["aggregate"])
                    else:
                        maskers = params.get_maskers1(next(iter(runner.explainers.values())),
                                                     include_legacy=False)

                        df = runner.evaluate_performance_drop1(maskers, 
                                                               use_last_time_only=True,
                                                               k_values = tuple(np.arange(argdict["k_min"], 
                                                                                         argdict["k_max"],
                                                                                         argdict["k_step"],))
                                                               )
                    log.info("Evaluating importance done.")

                    # Prepare the result dataframe to be saved.
                    df = df.reset_index()
                    original_columns = df.columns
                    now = datetime.now()
                    timestr = now.strftime("%Y%m%d-%H%M")
                    df["date"] = timestr
                    df["dataset"] = dataset.get_name()
                    df["explainer"] = next(iter(runner.explainers.values())).get_name()
                    df = df[["dataset", "explainer", "date"] + list(original_columns)]
                    all_df.append(df)
                    result_path = out_path / dataset.get_name()
                    result_path.mkdir(parents=True, exist_ok=True)
                    error_code = append_df_to_csv(df, result_path / result_file)
                    if first:
                        first = False
                    else:
                        if error_code != 0:
                            save_failed = False
        log.info(f"All done! Time elapsed: {time.time() - start_time}")
    except (Exception, KeyboardInterrupt) as e:
        if save_failed and len(all_df) > 0:
            result_file_bak = result_file + ".bak"
            pd.concat(all_df, axis=0, ignore_index=True).to_csv(str(out_path / result_file_bak),
                                                                index=False)
            log.info(f"Error! Emergency saved to {out_path / result_file_bak}")
        log.info(f"Time elapsed: {time.time() - start_time}")
        log.exception(e)
        raise e
