# -*- coding: utf-8 -*-

from os.path import join
from itertools import product
from string import digits

import numpy as np
import proplot as pplt
import pandas as pd

from vmbtd.datasets import (
    Article2DDip, Article2DFault, Article1DDZMax, Article1DFreq,
)
from vmbtd.architecture import (
    RCNN2DRegressor, Hyperparameters1D, Hyperparameters2D,
)
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample

TOINPUTS = ['shotgather']
TOOUTPUTS = ['vint']

params_1d = Hyperparameters1D(is_training=False)
params_1d.batch_size = 10
params_2d = Hyperparameters2D(is_training=False)
params_2d.batch_size = 10


class Models(Figure):
    @classmethod
    def construct(cls, name, datasets, log_subdir, is_2d, example_idx=0):
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        cls.is_2d = is_2d
        if not cls.is_2d:
            cls.params = params_1d
        else:
            cls.params = params_2d
        cls.datasets = {
            attribute: Dataset(cls.params)
            for attribute, Dataset in datasets.items()
        }
        cls.statistics = {
            (dataset, attribute): Statistics.construct(
                nn=RCNN2DRegressor,
                dataset=dataset,
                savedir=str(attribute),
            )
            for dataset in cls.datasets.values()
            for attribute in cls.datasets.keys()
        }
        cls.Metadata = CompoundMetadata.combine(
            *(
                Predictions.construct(
                    nn=RCNN2DRegressor,
                    params=cls.params,
                    logdir=join('logs', log_subdir, str(attribute)),
                    savedir=str(attribute),
                    dataset=dataset,
                    unique_suffix=str(attribute),
                )
                for dataset in cls.datasets.values()
                for attribute in cls.datasets.keys()
            ),
            *cls.statistics.values(),
            *(
                SelectExample.construct(
                    savedir=str(attribute),
                    dataset=dataset,
                    select=lambda s, m, filenames: filenames[example_idx],
                    SelectorMetadata=cls.statistics[(dataset, attribute)],
                )
                for dataset in cls.datasets.values()
                for attribute in cls.datasets.keys()
            ),
        )
        return cls

    def plot(self, data):
        dataset = next(iter(self.datasets.values()))
        attributes = self.datasets.keys()
        vint_meta = dataset.outputs['vint']

        fig, axs = pplt.subplots(
            nrows=4,
            ncols=4*2,
            figsize=[7.6, 4.5],
            sharey=1,
            sharex=True,
            wspace=([0, None]*4)[:-1],
        )

        table = pd.DataFrame(
            np.empty([4, 4]),
            index=[f'${a}$' for a in attributes],
            columns=[f'${a}$' for a in attributes],
            dtype=str,
        )
        name = dataset.name.rstrip(digits+'-')
        for source_dip, target_dip in product(attributes, attributes):
            key = f'Statistics_{name}{target_dip}_{source_dip}'
            statistics = data[key]
            if self.is_2d:
                metric = statistics['similarities']
                cell = f"${metric.mean():.3f} \\pm {metric.std():.3f}$"
            else:
                metric = statistics['rmses']
                cell = f"${round(metric.mean())} \\pm {round(metric.std())}$"
            table.loc[f'${source_dip}$', f'${target_dip}$'] = cell
        print(table.to_latex(escape=False))

        axs_pairs = [[axs[i], axs[i+1]] for i in range(0, len(axs), 2)]
        for (g_ax, p_ax), (value, dataset) in zip(
            axs_pairs, product(self.datasets.keys(), self.datasets.values()),
        ):
            example = data[f'SelectExample_{dataset.name}_{value}']
            label = example['labels/vint']
            pred = example['preds/vint']

            ref = example['labels/ref']
            crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0]*.95)

            vmin, vmax = dataset.model.properties['vp']
            for ax, im in zip([g_ax, p_ax], [label, pred]):
                im = im[crop_top:]
                try:
                    im, _ = vint_meta.postprocess(im)
                except AttributeError:
                    pass
                ax.imshow(
                    im/1000,
                    vmin=vmin/1000,
                    vmax=vmax/1000,
                    aspect='auto',
                    cmap='inferno',
                )

        dh = dataset.model.dh
        dt = dataset.acquire.dt * dataset.acquire.resampling
        tdelay = dataset.acquire.tdelay
        start_time = crop_top*dt - tdelay
        dcmp = dataset.acquire.ds * dh

        axs.format(
            abcloc='l',
            title="",
        )
        axs.format(
            ylabel="$t$ (s)",
            yscale=pplt.FuncScale(a=dt, b=start_time, decimals=1),
            xlabel="$x$ (km)",
            xscale=pplt.FuncScale(a=dcmp/1000, decimals=1),
        )
        for ax in axs[1::2]:
            ax.format(yticklabels=[])
        fig.colorbar(
            axs[0].images[0],
            label="$v_\\mathrm{int}(t, x)$ (km/s)",
            loc='r',
        )
        for ax in axs:
            ax.number = (ax.number-1)//2 + 1
        for i in range(0, axs.shape[1], 2):
            axs[:, i].format(abc='(a)')


Dips = Models.construct(
    'Dips', Article2DDip, 'dips', is_2d=True, example_idx=3,
)
Faults = Models.construct(
    'Faults', Article2DFault, 'faults', is_2d=True,
)
DZMaxs = Models.construct(
    'DZMaxs', Article1DDZMax, 'dzmaxs', is_2d=False,
)
DZMaxs.filename = 'dzmaxs.pdf'
Freqs = Models.construct(
    'Freqs', Article1DFreq, 'freqs', is_2d=False,
)

catalog.register(Dips)
catalog.register(Faults)
catalog.register(DZMaxs)
catalog.register(Freqs)
