# -*- coding: utf-8 -*-

from os.path import join
from itertools import product

import numpy as np
import proplot as pplt
import pandas as pd

from vmbtd.datasets import Article2DDip
from vmbtd.architecture import RCNN2DClassifier, Hyperparameters2D
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample

TOINPUTS = ['shotgather']
TOOUTPUTS = ['vint']

params = Hyperparameters2D(is_training=False)
params.batch_size = 10
dips = list(Article2DDip.keys())
datasets = {dip: Dataset(params) for dip, Dataset in Article2DDip.items()}
statistics = {
    (dataset, dip): Statistics.construct(
        nn=RCNN2DClassifier,
        dataset=dataset,
        savedir=str(dip),
    )
    for _, dataset in datasets.items()
    for dip, _ in datasets.items()
}


class Models(Figure):
    Metadata = CompoundMetadata.combine(
        *(
            Predictions.construct(
                nn=RCNN2DClassifier,
                params=params,
                logdir=join('logs', 'dips', str(dip)),
                savedir=str(dip),
                dataset=dataset,
                unique_suffix=str(dip),
            )
            for _, dataset in datasets.items()
            for dip, _ in datasets.items()
        ),
        *statistics.values(),
        *(
            SelectExample.construct(
                savedir=str(dip),
                dataset=dataset,
                select=SelectExample.partial_select_percentile(50),
                SelectorMetadata=statistics[(dataset, dip)],
            )
            for _, dataset in datasets.items()
            for dip, _ in datasets.items()
        ),
    )

    def plot(self, data):
        dataset = next(iter(datasets.values()))
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
            np.empty([4, 4]), index=dips, columns=dips, dtype=str,
        )
        for source_dip, target_dip in product(dips, dips):
            key = f'Statistics_Article2DDip{target_dip}_{source_dip}'
            statistics = data[key]
            metric = statistics['similarities']
            cell = f"${metric.mean():.3f} \\pm {metric.std():.3f}$"
            table.loc[source_dip, target_dip] = cell
        print(table.to_latex(escape=False))

        axs_pairs = [[axs[i], axs[i+1]] for i in range(0, len(axs), 2)]
        for (g_ax, p_ax), (dip, dataset) in zip(
            axs_pairs, product(datasets.keys(), datasets.values()),
        ):
            example = data[f'SelectExample_{dataset.name}_{dip}']
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


catalog.register(Models)
