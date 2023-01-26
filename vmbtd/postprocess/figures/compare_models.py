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
    RCNN2DClassifier as RCNN2D, Hyperparameters1D, Hyperparameters2D,
)
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample

TOINPUTS = ['shotgather']
TOOUTPUTS = ['vint']

params_1d = Hyperparameters1D(is_training=False)
params_1d.batch_size = 10
params_2d = Hyperparameters2D(is_training=False)
params_2d.batch_size = 2


class Models1D(Figure):
    params = params_1d

    @classmethod
    def construct(
        cls, name, datasets, log_subdir,
        attribute_symbol, attribute_units, example_idx=0,
    ):
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        cls.attribute_symbol = attribute_symbol
        cls.attribute_units = attribute_units
        cls.datasets = {
            attribute: Dataset(cls.params)
            for attribute, Dataset in datasets.items()
        }
        cls.statistics = {
            (dataset, attribute): Statistics.construct(
                nn=RCNN2D,
                dataset=dataset,
                savedir=str(attribute),
            )
            for dataset in cls.datasets.values()
            for attribute in cls.datasets.keys()
        }
        cls.Metadata = CompoundMetadata.combine(
            *(
                Predictions.construct(
                    nn=RCNN2D,
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

    @property
    def attributes(self):
        attributes = self.datasets.keys()
        return attributes, attributes

    def plot(self, data):
        self.dataset = dataset = next(iter(self.datasets.values()))
        train_att, test_att = self.attributes
        vint_meta = dataset.outputs['vint']

        fig, axs = self.subplots()

        table = pd.DataFrame(
            np.empty([len(train_att), len(test_att)]),
            dtype=str,
        )
        name = dataset.name.rstrip(digits+'-')
        for (i, source_att), (j, target_att) in product(
            enumerate(train_att), enumerate(test_att),
        ):
            key = f'Statistics_{name}{target_att}_{source_att}'
            statistics = data[key]
            metric = statistics['rmses']
            cell = f"${round(metric.mean())} \\pm {round(metric.std())}$"
            table.loc[i, j] = cell
        table = self.format_table(table)
        print(table.to_latex(escape=False))

        axs_pairs = self.pair(axs)
        ymax = np.inf
        for (g_ax, p_ax), (value, dataset) in zip(
            axs_pairs, product(train_att, self.datasets.values()),
        ):
            example = data[f'SelectExample_{dataset.name}_{value}']
            label = example['labels/vint']
            pred = example['preds/vint']

            ref = example['labels/ref']
            crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0]*.95)

            for ax, im in zip([g_ax, p_ax], [label, pred]):
                im = im[crop_top:]
                try:
                    im, _ = vint_meta.postprocess(im)
                except AttributeError:
                    pass
                ymax = len(im) if len(im) < ymax else ymax
                self.imshow(ax, im)

        self.format(fig, axs, crop_top, ymax)
        self.suptitles(fig)

    def subplots(self):
        train_att, test_att = self.attributes
        nrows = len(train_att)
        ncols = len(test_att)
        return pplt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=[3.3, 5],
            share=True,
            wspace=None,
            left=8.5,
            top=5.5,
        )

    def format_table(self, table):
        train_att, test_att = self.attributes
        table.index = [f'${abs(a)}$' for a in train_att]
        table.columns = [f'${abs(a)}$' for a in test_att]
        return table

    def pair(self, axs):
        return [[axs[i], axs[i]] for i in range(0, len(axs), 1)]

    def imshow(self, ax, im):
        ax.plot(
            im.flatten()/1000,
            range(len(im)),
        )

    def format(self, fig, axs, crop_top, ymax):
        dt = self.dataset.acquire.dt * self.dataset.acquire.resampling
        tdelay = self.dataset.acquire.tdelay
        start_time = crop_top*dt - tdelay
        vmin, vmax = self.dataset.model.properties['vp']

        units = self.attribute_units
        axs.format(
            abcloc='l',
            title="",
            yscale=pplt.FuncScale(a=dt, b=start_time, decimals=1),
        )
        att, _ = self.attributes
        label_attributes = [f'${abs(a)}{units}$' for a in att]

        axs.format(
            abc='(a)',
            ylim=[ymax, 0],
            xlim=[vmin/1000, vmax/1000],
            leftlabels=label_attributes,
            toplabels=label_attributes,
        )
        fig.legend(
            handles=axs[0].get_lines(),
            labels=['Cible', 'Prédiction'],
            loc='b'
        )

        axs[0, 0].set_ylabel("$t$ (s)")
        axs[-1, 0].set_xlabel("$V(t)$\n(km/s)")
        for ax in axs[1:, :]:
            ax.format(yticklabels=[])
        for ax in axs[:, 1:]:
            ax.format(xticklabels=[])

    def suptitles(self, fig):
        symbol = self.attribute_symbol
        x_top, y_top, x_left, y_left = self.get_suptitles_loc()
        fig.text(
            x=x_top,
            y=y_top,
            s=f"${symbol}_\\mathrm{{test}}$",
            fontweight='bold',
            fontsize='large',
            ha='center',
            va='bottom',
        )
        fig.text(
            x=x_left,
            y=y_left,
            s=f"${symbol}_\\mathrm{{ent.}}$",
            fontweight='bold',
            fontsize='large',
            rotation=90,
            ha='right',
            va='center',
        )

    def get_suptitles_loc(self):
        # x_top, y_top, x_left, y_left
        return .63, .96, .07, .53


class Models2D(Models1D):
    params = params_2d

    def subplots(self):
        train_att, test_att = self.attributes
        nrows = len(train_att)
        ncols = len(test_att)
        return pplt.subplots(
            nrows=nrows,
            ncols=ncols * 2,
            figsize=[7.6, 4.5],
            share=True,
            wspace=([0, None]*3)[:-1],
            left=8.5,
            top=5.5,
        )

    def pair(self, axs):
        return [[axs[i], axs[i+1]] for i in range(0, len(axs), 2)]

    def imshow(self, ax, im):
        vmin, vmax = self.dataset.model.properties['vp']
        ax.imshow(
            im/1000,
            vmin=vmin/1000,
            vmax=vmax/1000,
            aspect='auto',
            cmap='inferno',
        )

    def format(self, fig, axs, crop_top, ymax):
        dh = self.dataset.model.dh
        dt = self.dataset.acquire.dt * self.dataset.acquire.resampling
        tdelay = self.dataset.acquire.tdelay
        start_time = crop_top*dt - tdelay
        dcmp = self.dataset.acquire.ds * dh

        units = self.attribute_units
        att, _ = self.attributes
        label_attributes = [f'${abs(a)}{units}$' for a in att]

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

        axs[0, 0].set_ylabel("$t$ (s)")
        axs[-1, 0].set_xlabel("$x$ (km)")
        for ax in axs[1:, :]:
            ax.format(yticklabels=[])
        for ax in axs[:, 1:]:
            ax.format(xticklabels=[])

    def get_suptitles_loc(self):
        # x_top, y_top, x_left, y_left
        return .5, .96, .03, .5


DZMaxs = Models1D.construct(
    'DZMaxs', Article1DDZMax, 'dzmaxs', example_idx=0,
    attribute_symbol='\\Delta{V}', attribute_units='~\\mathrm{m/s}'
)
DZMaxs.filename = 'dzmaxs.pdf'
Freqs = Models1D.construct(
    'Freqs', Article1DFreq, 'freqs', example_idx=0,
    attribute_symbol='f', attribute_units='~\\mathrm{Hz}'
)
DIPS_EXAMPLE_IDX = 3
Dips = Models2D.construct(
    'Dips', Article2DDip, 'dips', example_idx=DIPS_EXAMPLE_IDX,
    attribute_symbol='\\theta', attribute_units='^\\circ'
)
Faults = Models2D.construct(
    'Faults', Article2DFault, 'faults', example_idx=DIPS_EXAMPLE_IDX,
    attribute_symbol='d', attribute_units='~\\mathrm{m}'
)

catalog.register(DZMaxs)
catalog.register(Freqs)
catalog.register(Dips)
catalog.register(Faults)
