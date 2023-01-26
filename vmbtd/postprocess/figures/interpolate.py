# -*- coding: utf-8 -*-

from os import listdir
from os.path import join, exists

import numpy as np
import pandas as pd
import proplot as pplt
from matplotlib.colors import TABLEAU_COLORS

from vmbtd.datasets import Article2DDip, InterpolateDips
from vmbtd.architecture import (
    RCNN2DClassifier as RCNN2D, Hyperparameters2DInterpolate as Params,
)
from ..catalog import catalog, Figure, Metadata, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample
from .compare_models import Models2D, DIPS_EXAMPLE_IDX

TABLEAU_COLORS = list(TABLEAU_COLORS)

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


class LossInterpolate_(Metadata):
    logdirs = {'interpolate': join('logs', 'interpolate')}
    columns = ['vint_loss']

    def generate(self, gpus):
        for nn, logdir in self.logdirs.items():
            sublogdirs = [logdir]
            mean, std = self.load_all_events(sublogdirs)
            for item, value in zip(['mean', 'std'], [mean, std]):
                key = nn + '/' + item
                self[key] = value

    def load_all_events(self, logdirs):
        data = []
        for logdir in logdirs:
            current_data = self.load_events(logdir)
            data.append(current_data)
        data = pd.concat(data)
        data = data[self.columns]
        by_index = data.groupby(data.index)
        return by_index.mean(), by_index.std()

    def load_events(self, logdir):
        events_path = join(logdir, 'progress.csv')
        assert exists(events_path)
        return pd.read_csv(events_path)


class LossInterpolate(Figure):
    Metadata = LossInterpolate_

    def plot(self, data):
        _, (loss_ax, prop_ax) = pplt.subplots(
            nrows=2,
            height_ratios=[2, 1],
            figsize=[3.33, 4],
            sharex=True,
            sharey=False,
        )
        epochs = np.arange(params.epochs) + 1
        LABELS = ['vint_loss']

        for nn, ls in zip(self.Metadata.logdirs.keys(), ['-', '--']):
            mean = data[nn + '/mean']
            mean = pd.DataFrame(mean, columns=self.Metadata.columns)
            std = data[nn + '/std']
            std = pd.DataFrame(std, columns=self.Metadata.columns)

            for column in mean.columns:
                if column not in LABELS:
                    del mean[column]
                    del std[column]
            for i, column in enumerate(LABELS):
                current_mean = mean[column]
                loss_ax.plot(
                    epochs,
                    current_mean,
                    c='k',
                )
        loss_ax.format(
            xlabel="Époque",
            ylabel="$L",
        )

        dataset_interpolate = InterpolateDips(params)
        dataset_interpolate.tfdataset(
            phase="train",
            tooutputs=['vint'],
            toinputs=['shotgather'],
            batch_size=params.batch_size,
        )
        proportions = []
        for epoch in epochs:
            proportions.append(dataset_interpolate.proportions)
            try:
                dataset_interpolate.on_epoch_end()
            except ValueError:
                pass
        proportions = np.array(proportions).T
        bottom = 0
        for dip, prop in zip(dataset_interpolate.subsets.keys(), proportions):
            top = bottom + prop
            prop_ax.fill_between(epochs, bottom, top, label=f'${dip}^\\circ$')
            bottom = top
        prop_ax.format(
            ylim=[0, 200],
            ylabel="Quantité\nde modèles",
            grid=False,
            gridminor=False,
        )
        prop_ax.legend(ncols=1)


class Interpolate(Models2D):
    datasets = datasets
    attribute_symbol = '\\theta'
    attribute_units = '^\\circ'

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

    @property
    def attributes(self):
        attributes = self.datasets.keys()
        return ['interpolate'], attributes

    def subplots(self):
        _, test_att = self.attributes
        ncols = len(test_att)
        return pplt.subplots(
            nrows=1,
            ncols=ncols * 2,
            figsize=[7.6, 1.8],
            share=True,
            wspace=([0, None]*3)[:-1],
            top=5.5,
        )

    def format_table(self, table):
        _, test_att = self.attributes
        table.index = ['']
        table.columns = [f'${abs(a)}$' for a in test_att]
        return table

    def format(self, fig, axs, crop_top, ymax):
        dh = self.dataset.model.dh
        dt = self.dataset.acquire.dt * self.dataset.acquire.resampling
        tdelay = self.dataset.acquire.tdelay
        start_time = crop_top*dt - tdelay
        dcmp = self.dataset.acquire.ds * dh

        units = self.attribute_units
        _, att = self.attributes
        label_attributes = [f'${abs(a)}{units}$' for a in att]

        axs[0, 0].set_ylabel("$t$ (s)")

        axs.format(
            abcloc='l',
            title="",
            yscale=pplt.FuncScale(a=dt, b=start_time, decimals=1),
            ylim=[ymax, 0],
            xscale=pplt.FuncScale(a=dcmp/1000, decimals=1),
            xlocator=125/dh,
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
            label="$V(t, x)$ (km/s)",
            loc='r',
        )
        for ax in axs:
            ax.number = (ax.number-1)//2 + 1
        for i in range(0, axs.shape[1], 2):
            axs[:, i].format(abc='(a)')
        for ax in axs[:, 1:]:
            ax.format(xticklabels=[])

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
        return .5, .88, None, None


catalog.register(LossInterpolate)
catalog.register(Interpolate)
