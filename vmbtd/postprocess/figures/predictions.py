# -*- coding: utf-8 -*-

from os import listdir, makedirs
from os.path import join, exists, split
from datetime import datetime
from argparse import Namespace
from functools import partial

import numpy as np
from skimage.measure import compare_ssim as ssim

from vmbtd.__main__ import main as global_main
from ..catalog import Metadata

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']


class Predictions(Metadata):
    @classmethod
    def construct(
        cls, *, nn, params, logdir, savedir, dataset,
        do_generate_dataset=False, unique_suffix=None,
    ):
        name = '_'.join([cls.__name__, type(dataset).__name__, savedir])
        if unique_suffix is not None:
            name = name + '_' + unique_suffix
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        cls.nn = nn
        cls.params = params
        cls.logdir = logdir
        if savedir is None:
            savedir = nn.__name__
        cls.savedir = savedir
        cls.dataset = dataset
        cls.generate_dataset = do_generate_dataset
        return cls

    def generate(self, gpus):
        nn = self.nn
        params = self.params
        logdir = self.logdir
        savedir = self.savedir
        dataset = self.dataset
        dataset._getfilelist()

        print("Launching inference.")
        print("NN:", nn.__name__)
        print("Hyperparameters:", type(params).__name__)
        print("Weights:", logdir)
        print("Case:", savedir)

        trigger_predictions = bool(gpus)
        if trigger_predictions or not exists(join(dataset.datatest, savedir)):
            logdirs = listdir(logdir)
            if any('checkpoint' in dir for dir in logdirs):
                logdirs = ['']
            start_times = []
            for i, current_logdir in enumerate(logdirs):
                print(f"Using NN {i+1} out of {len(logdirs)}.")
                start_times.append(str(datetime.now()))
                print(f"Started at {start_times[-1]}.")
                current_logdir = join(logdir, current_logdir)
                current_savedir = f"{savedir}_{i}"
                current_args = Namespace(
                    nn=nn,
                    params=params,
                    dataset=dataset,
                    logdir=current_logdir,
                    generate=self.generate_dataset,
                    train=False,
                    test=True,
                    gpus=gpus,
                    savedir=current_savedir,
                    plot=False,
                    debug=False,
                    eager=False,
                    workdirs=None,
                )
                global_main(current_args)
            end_time = str(datetime.now())
            print(f"Finished at {end_time}.")

        combine_predictions(dataset, logdir, savedir)


class SelectExample(Metadata):
    COLNAMES = ['inputs', 'preds', 'std', 'labels', 'weights']

    savedir = None
    dataset = None
    SelectorMetadata = None

    @classmethod
    def construct(
        cls, *, savedir, dataset, select, SelectorMetadata, unique_suffix=None,
    ):
        name = '_'.join([cls.__name__, type(dataset).__name__, savedir])
        if unique_suffix is not None:
            name = name + '_' + unique_suffix
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        cls.savedir = savedir
        cls.dataset = dataset
        cls.select = select
        cls.SelectorMetadata = SelectorMetadata
        return cls

    def generate(self, _):
        dataset = self.dataset
        savedir = self.savedir

        metadata = self.SelectorMetadata(None)
        dataset._getfilelist()
        filenames = dataset.files['test']
        filename = self.select(metadata, filenames)

        inputs, labels, weights, filename = dataset.get_example(
            filename=filename,
            phase='test',
            toinputs=TOINPUTS,
            tooutputs=TOOUTPUTS,
        )

        preds = dataset.generator.read_predictions(filename, savedir)
        preds = {name: preds[name] for name in TOOUTPUTS}
        if exists(join(dataset.datatest, savedir + "_std")):
            std = dataset.generator.read_predictions(
                filename, savedir + "_std",
            )
        elif any(preds[key].shape[-2] > 1 for key in TOOUTPUTS):
            std = {
                key: dataset.outputs[key].postprocess(preds[key])[1]
                for key in TOOUTPUTS
            }
        else:
            std = {}
        cols = [inputs, preds, std, labels, weights]

        for colname, col in zip(self.COLNAMES, cols):
            for item, value in col.items():
                key = colname + '/' + item
                self[key] = value

    def select(self, metadata, filenames):
        raise NotImplementedError

    def use_select_percentile(self, percentile):
        self.select = self.partial_select_percentile(percentile)

    @staticmethod
    def partial_select_percentile(percentile):
        return partial(SelectExample.select_percentile, percentile=percentile)

    @staticmethod
    def select_percentile(metadata, filenames, percentile):
        if metadata['similarities'].any():
            metric = metadata['similarities']
        else:
            metric = metadata['rmses']

        score = np.percentile(metric, percentile, interpolation="nearest")
        idx = np.argwhere(score == metric)[0, 0]
        print(f"{percentile}th percentile: {score} for example {idx}.")
        return filenames[idx]


def combine_predictions(dataset, logdir, savedir):
    print("Averaging predictions.")
    logdirs = listdir(logdir)
    if any('checkpoint' in dir for dir in logdirs):
        logdirs = ['']
    dataset._getfilelist()
    for filename in dataset.files["test"]:
        preds = {key: [] for key in dataset.generator.outputs}
        for i in range(len(logdirs)):
            current_load_dir = f"{savedir}_{i}"
            current_preds = dataset.generator.read_predictions(
                filename, current_load_dir,
            )
            for key, value in current_preds.items():
                preds[key].append(value)
        average = {
            key: np.mean(value, axis=0) for key, value in preds.items()
        }
        std = {
            key: np.std(value, axis=0) for key, value in preds.items()
        }
        directory, filename = split(filename)
        filedir = join(directory, savedir)
        if not exists(filedir):
            makedirs(filedir)
        dataset.generator.write_predictions(
            None, filedir, average, filename=filename,
        )
        if not exists(f"{filedir}_std"):
            makedirs(f"{filedir}_std")
        dataset.generator.write_predictions(
            None, f"{filedir}_std", std, filename=filename,
        )


class Statistics(Metadata):
    colnames = ['similarities', 'rmses']

    @classmethod
    def construct(cls, *, nn, dataset, savedir, unique_suffix=None):
        name = '_'.join([cls.__name__, type(dataset).__name__, savedir])
        if unique_suffix is not None:
            name = name + '_' + unique_suffix
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        cls.nn = nn
        if savedir is None:
            savedir = nn.__name__
        cls.savedir = savedir
        cls.dataset = dataset
        return cls

    def generate(self, _):
        savedir = self.savedir
        dataset = self.dataset
        meta_output = dataset.outputs['vint']

        print(
            f"Comparing predictions for dataset {dataset.name}, "
            f"directory {savedir}."
        )
        _, labels, weights, preds = read_all(dataset, savedir)

        similarities = np.array([])
        rmses = np.array([])
        for current_labels, current_weights, current_preds in zip(
            labels["vint"], weights["vint"], preds["vint"],
        ):
            current_labels, _ = meta_output.reduce(current_labels)
            current_labels *= current_weights
            current_preds, _ = meta_output.reduce(current_preds)
            current_preds *= current_weights
            if current_labels.shape[1] != 1:
                similarity = ssim(current_labels, current_preds)
                similarities = np.append(similarities, similarity)
            rmse = np.sqrt(np.mean((current_labels-current_preds)**2))
            rmses = np.append(rmses, rmse)
        vmin, vmax = dataset.model.properties['vp']
        rmses *= vmax - vmin

        metrics = [similarities, rmses]
        for name, metric in zip(self.colnames, metrics):
            self[name] = metric

    def print_statistics(self):
        similarities = self['similarities']
        rmses = self['rmses']
        print(
            f"Statistics for dataset {self.dataset.name}, "
            f"directory {self.savedir}."
        )
        print("Average SSIM:", np.mean(similarities))
        print("Standard deviation on SSIM:", np.std(similarities))
        print("Average RMSE:", np.mean(rmses))
        print("Standard deviation on RMSE:", np.std(rmses))


def read_all(dataset, savedir=None, toinputs=TOINPUTS, tooutputs=TOOUTPUTS):
    all_inputs = {}
    all_labels = {}
    all_weights = {}
    all_preds = {}
    dataset._getfilelist()
    for example in dataset.files["test"]:
        inputs, labels, weights, _ = dataset.get_example(
            example,
            phase='test',
            toinputs=toinputs,
            tooutputs=tooutputs,
        )
        if savedir is not None:
            preds = dataset.generator.read_predictions(example, savedir)
        else:
            preds = {}
        target_dicts = [all_inputs, all_labels, all_weights, all_preds]
        current_dicts = [inputs, labels, weights, preds]
        for target_dict, current_dict in zip(target_dicts, current_dicts):
            for key in current_dict.keys():
                current_array = np.expand_dims(current_dict[key], axis=0)
                if key in target_dict.keys():
                    target_dict[key] = np.append(
                        target_dict[key], current_array, axis=0,
                    )
                else:
                    target_dict[key] = current_array
    return all_inputs, all_labels, all_weights, all_preds
