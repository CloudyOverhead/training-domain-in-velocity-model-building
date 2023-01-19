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


class Models(Figure):
    @classmethod
    def construct(
        cls, name, datasets, log_subdir, is_2d,
        attribute_symbol, attribute_units, example_idx=0,
    ):
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        cls.is_2d = is_2d
        cls.attribute_symbol = attribute_symbol
        cls.attribute_units = attribute_units
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

    def plot(self, data):
        dataset = next(iter(self.datasets.values()))
        attributes = self.datasets.keys()
        vint_meta = dataset.outputs['vint']

        if self.is_2d:
            ncols = 4 * 2
            figsize = [7.6, 4.5]
            wspace = ([0, None]*4)[:-1]
        else:
            ncols = 4
            figsize = [3.3, 5]
            wspace = None
        fig, axs = pplt.subplots(
            nrows=4,
            ncols=ncols,
            figsize=figsize,
            share=True,
            wspace=wspace,
            left=8.5,
            top=5,
        )

        table = pd.DataFrame(
            np.empty([4, 4]),
            index=[f'${abs(a)}$' for a in attributes],
            columns=[f'${abs(a)}$' for a in attributes],
            dtype=str,
        )
        name = dataset.name.rstrip(digits+'-')
        for source_att, target_att in product(attributes, attributes):
            key = f'Statistics_{name}{target_att}_{source_att}'
            statistics = data[key]
            metric = statistics['rmses']
            cell = f"${round(metric.mean())} \\pm {round(metric.std())}$"
            table.loc[f'${abs(source_att)}$', f'${abs(target_att)}$'] = cell
        print(table.to_latex(escape=False))

        if self.is_2d:
            axs_pairs = [[axs[i], axs[i+1]] for i in range(0, len(axs), 2)]
        else:
            axs_pairs = [[axs[i], axs[i]] for i in range(0, len(axs), 1)]
        ymax = np.inf
        vmin, vmax = dataset.model.properties['vp']
        for (g_ax, p_ax), (value, dataset) in zip(
            axs_pairs, product(self.datasets.keys(), self.datasets.values()),
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
                if self.is_2d:
                    ax.imshow(
                        im/1000,
                        vmin=vmin/1000,
                        vmax=vmax/1000,
                        aspect='auto',
                        cmap='inferno',
                    )
                else:
                    ax.plot(
                        im.flatten()/1000,
                        range(len(im)),
                    )

        dh = dataset.model.dh
        dt = dataset.acquire.dt * dataset.acquire.resampling
        tdelay = dataset.acquire.tdelay
        start_time = crop_top*dt - tdelay
        dcmp = dataset.acquire.ds * dh

        symbol = self.attribute_symbol
        units = self.attribute_units
        axs.format(
            abcloc='l',
            title="",
            yscale=pplt.FuncScale(a=dt, b=start_time, decimals=1),
        )
        label_attributes = [f'${abs(a)}{units}$' for a in attributes]
        if self.is_2d:
            axs.format(
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
        else:
            axs[-1, 0].set_xlabel("$v_\\mathrm{int}(t)$\n(km/s)")
            axs.format(
                ylim=[0, ymax],
                yreverse=True,
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
        for ax in axs[1:, :]:
            ax.format(yticklabels=[])
        for ax in axs[:, 1:]:
            ax.format(xticklabels=[])

        fig.text(
            x=.5 if self.is_2d else .63,
            y=.95 if self.is_2d else .95,
            s=f"${symbol}_\\mathrm{{ent.}}$",
            fontweight='bold',
            fontsize='large',
            ha='center',
            va='bottom',
        )
        fig.text(
            x=.03 if self.is_2d else .06,
            y=.5 if self.is_2d else .53,
            s=f"${symbol}_\\mathrm{{test}}$",
            fontweight='bold',
            fontsize='large',
            rotation=90,
            ha='right',
            va='center',
        )


Dips = Models.construct(
    'Dips', Article2DDip, 'dips', is_2d=True, example_idx=3,
    attribute_symbol='\\theta', attribute_units='^\\circ'
)
Faults = Models.construct(
    'Faults', Article2DFault, 'faults', is_2d=True,
    attribute_symbol='d', attribute_units='~\\mathrm{m}'
)
DZMaxs = Models.construct(
    'DZMaxs', Article1DDZMax, 'dzmaxs', is_2d=False, example_idx=0,
    attribute_symbol='\\Delta{V}', attribute_units='~\\mathrm{m/s}'
)
DZMaxs.filename = 'dzmaxs.pdf'
Freqs = Models.construct(
    'Freqs', Article1DFreq, 'freqs', is_2d=False,
    attribute_symbol='f', attribute_units='~\\mathrm{Hz}'
)

catalog.register(Dips)
catalog.register(Faults)
catalog.register(DZMaxs)
catalog.register(Freqs)
