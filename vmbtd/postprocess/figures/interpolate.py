# -*- coding: utf-8 -*-

from os.path import join

import proplot as pplt

from vmbtd.datasets import Article2DDip
from vmbtd.architecture import (
    RCNN2DClassifier as RCNN2D, Hyperparameters2DInterpolate as Params,
)
from ..catalog import catalog, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample
from .compare_models import Models2D, DIPS_EXAMPLE_IDX

TOINPUTS = ['shotgather']
TOOUTPUTS = ['vint']

SAVEDIR = 'interpolate'


params = Params(is_training=False)
params.batch_size = 2
datasets = {
    attribute: Dataset(params)
    for attribute, Dataset in Article2DDip.items()
}
statistics = {
    dataset: Statistics.construct(
        nn=RCNN2D,
        dataset=dataset,
        savedir=SAVEDIR,
    )
    for dataset in datasets.values()
}
attribute_symbol = '\\theta'
attribute_units = '^\\circ'


class InterpolateDips(Models2D):
    Metadata = CompoundMetadata.combine(
        *(
            Predictions.construct(
                nn=RCNN2D,
                params=params,
                logdir=join('logs', 'interpolate'),
                savedir=SAVEDIR,
                dataset=dataset,
                unique_suffix='interpolate',
            )
            for dataset in datasets.values()
        ),
        *statistics.values(),
        *(
            SelectExample.construct(
                savedir=SAVEDIR,
                dataset=dataset,
                select=lambda s, m, filenames: filenames[DIPS_EXAMPLE_IDX],
                SelectorMetadata=statistics[dataset],
            )
            for dataset in datasets.values()
        ),
    )

    def subplots(self, attributes):
        nrows = len(attributes)
        return pplt.subplots(
            nrows=1,
            ncols=nrows * 2,
            figsize=[7.6, 4.5],
            share=True,
            wspace=([0, None]*3)[:-1],
            left=8.5,
            top=5.5,
        )

    def pair(self, axs):
        return [[axs[i], axs[i+1]] for i in range(0, len(axs), 2)]

    def format(self, fig, axs, attributes, crop_top, ymax):
        dh = self.dataset.model.dh
        dt = self.dataset.acquire.dt * self.dataset.acquire.resampling
        tdelay = self.dataset.acquire.tdelay
        start_time = crop_top*dt - tdelay
        dcmp = self.dataset.acquire.ds * dh

        units = self.attribute_units
        label_attributes = [f'${abs(a)}{units}$' for a in attributes]

        axs[0, 0].set_ylabel("$t$ (s)")
        for ax in axs[1:, :]:
            ax.format(yticklabels=[])
        for ax in axs[:, 1:]:
            ax.format(xticklabels=[])

        axs.format(
            abcloc='l',
            title="",
            yscale=pplt.FuncScale(a=dt, b=start_time, decimals=1),
            ylim=[ymax, 0],
            xscale=pplt.FuncScale(a=dcmp/1000, decimals=1),
            xlocator=125/dh,
            leftlabels=label_attributes,
            suptitlepad=.50,
        )
        axs[-1, 0].set_xlabel("$x$ (km)")
        for ax, attribute in zip(
            [axs[0, i] for i in range(0, axs.shape[1], 2)],
            label_attributes,
        ):
            ax.text(
                x=1,
                y=1.25,
                s=attribute,
                transform='axes',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize='med-large',
            )
        fig.colorbar(
            axs[0].images[0],
            label="$v_\\mathrm{int}(t, x)$ (km/s)",
            loc='r',
        )
        for ax in axs:
            ax.number = (ax.number-1)//2 + 1
        for i in range(0, axs.shape[1], 2):
            axs[:, i].format(abc='(a)')

    def suptitles(self, fig):
        symbol = self.attribute_symbol
        x_top, y_top, _, _ = self.get_suptitles_loc()
        fig.text(
            x=x_top,
            y=y_top,
            s=f"${symbol}_\\mathrm{{test}}$",
            fontweight='bold',
            fontsize='large',
            ha='center',
            va='bottom',
        )

    def get_suptitles_loc(self):
        # x_top, y_top, x_left, y_left
        return .5, .96, None, None


catalog.register(InterpolateDips)
