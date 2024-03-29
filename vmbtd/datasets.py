# -*- coding: utf-8 -*-

from os import makedirs
from os.path import abspath, join
from glob import glob
from itertools import cycle, chain
from copy import copy
from shutil import copy as fcopy

import numpy as np
from ModelGenerator import (
    Sequence as GeoSequence, Stratigraphy, Deformation, Property, Lithology,
)
from tensorflow.keras.utils import Sequence
from tensorflow.python.data.util import options as options_lib
from tensorflow.data.experimental import DistributeOptions, AutoShardPolicy
from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import (
    Reftime, Vrms, Vint, Vdepth, ShotGather,
)
from natsort import natsorted as sorted



DistributeOptions.auto_shard_policy = options_lib.create_option(
    name="auto_shard_policy",
    ty=AutoShardPolicy,
    docstring="The type of sharding to use. See "
              "`tf.data.experimental.AutoShardPolicy` for additional "
              "information.",
    default_factory=lambda: AutoShardPolicy.DATA,
)


class Dataset(GeoDataset, Sequence):
    basepath = abspath("datasets")

    def __init__(self, params, noise=False):
        self.params = params
        super().__init__()
        if noise:
            for input in self.inputs.values():
                input.random_static = True
                input.random_static_max = 1
                input.random_noise = True
                input.random_noise_max = 0.02
                input.random_time_scaling = True

    def get_example(
        self, filename=None, phase="train", shuffle=True, toinputs=None,
        tooutputs=None,
    ):
        if tooutputs is None:
            tooutputs = list(self.outputs.keys())

        if filename is None:
            do_reset_iterator = (
                not hasattr(self, "iter_examples")
                or not self.files[self.phase]
            )
            if do_reset_iterator:
                self.tfdataset(phase, shuffle, tooutputs, toinputs)
            filename = next(self.iter_examples)

        inputs, labels, weights, filename = super().get_example(
            filename, phase, shuffle, toinputs, tooutputs,
        )
        return inputs, labels, weights, filename

    def tfdataset(
        self, phase="train", shuffle=True, tooutputs=None, toinputs=None,
        batch_size=1,
    ):
        if phase == "validate" and self.validatesize == 0:
            return
        self.phase = phase

        self.shuffle = shuffle
        self.tooutputs = tooutputs
        self.toinputs = toinputs
        self.batch_size = batch_size

        phases = {
            "train": self.datatrain,
            "validate": self.datavalidate,
            "test": self.datatest,
        }
        pathstr = join(phases[phase], 'example_*')
        self.files[self.phase] = sorted(glob(pathstr))

        if shuffle:
            np.random.shuffle(self.files[self.phase])
        self.iter_examples = cycle(self.files[self.phase])

        self.on_epoch_end()

        return copy(self)

    def __getitem__(self, idx):
        batch = self.batches_idx[idx]
        data = {in_: [] for in_ in self.toinputs}
        data['filename'] = []
        labels = {out: [] for out in self.tooutputs}
        for i in batch:
            filename = self.files[self.phase][i]
            data_i, labels_i, weights_i, _ = self.get_example(
                filename=filename,
                toinputs=self.toinputs,
                tooutputs=self.tooutputs,
            )
            for in_ in self.toinputs:
                data[in_].append(data_i[in_])
            data["filename"].append([filename])
            for out in self.tooutputs:
                labels[out].append([labels_i[out], weights_i[out]])
        for key, value in data.items():
            data[key] = np.array(value)
        for key, value in labels.items():
            labels[key] = np.array(value)
        return data, labels

    def __len__(self):
        return int(len(self.files[self.phase]) / self.batch_size)

    def on_epoch_end(self):
        self.batches_idx = np.arange(len(self) * self.batch_size)
        if self.shuffle:
            self.batches_idx = np.random.choice(
                self.batches_idx,
                [len(self), self.batch_size],
                replace=False,
            )
        else:
            self.batches_idx = self.batches_idx.reshape(
                [len(self), self.batch_size]
            )


class Article1D(Dataset):
    dzmax = 1000
    peak_freq = 25
    dip_max = 1E-2
    fault_displ_min = 0
    fault_prob = 0
    dh = 6.25

    @classmethod
    def construct(cls, suffix='', **attributes):
        name = f"{cls.__name__}{suffix}"
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        for attribute_name, value in attributes.items():
            if attribute_name == 'dip_max' and value < 1E-2:
                value = 1E-2
            setattr(cls, attribute_name, value)
        return cls

    def set_dataset(self):
        self.trainsize = 2000
        self.validatesize = 0
        self.testsize = 100

        model = MarineModel()
        model.dh = self.dh
        model.NX = 692 * 2
        model.NZ = 752 * 2
        model.layer_num_min = 48
        model.layer_dh_min = 50
        model.layer_dh_max = 200
        model.water_vmin = 1430
        model.water_vmax = 1560
        model.water_dmin = .9 * model.water_vmin
        model.water_dmax = 3.1 * model.water_vmax
        model.vp_min = 1300.0
        model.vp_max = 4000.0
        model.dzmin = None
        model.dzmax = self.dzmax
        model.accept_decrease = .65
        model.dip_max = self.dip_max

        acquire = Acquisition(model=model)
        acquire.dt = .0004
        acquire.NT = int(8 / acquire.dt)
        acquire.resampling = 10
        acquire.dg = 8
        acquire.ds = 8
        acquire.gmin = gmin = int(470 / model.dh)
        acquire.gmax = gmax = int((470+72*acquire.dg*model.dh) / model.dh)
        acquire.minoffset = 470
        acquire.peak_freq = self.peak_freq
        acquire.df = 5
        acquire.wavefuns = [0, 1]
        acquire.source_depth = (acquire.Npad+4) * model.dh
        acquire.receiver_depth = (acquire.Npad+4) * model.dh
        acquire.tdelay = 3.0 / (25-5)
        acquire.singleshot = True
        acquire.configuration = 'inline'

        npad = acquire.Npad
        model.fault_dip_min = model.fault_dip_max = 90
        model.fault_displ_min = self.fault_displ_min
        model.fault_x_lim = [gmax//2+npad, model.NX-gmax-gmin//2-npad]
        model.fault_y_lim = [0, model.NZ]
        model.fault_prob = self.fault_prob

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        bins = self.params.decode_bins
        outputs = {
            Reftime.name: Reftime(model=model, acquire=acquire),
            Vrms.name: Vrms(model=model, acquire=acquire, bins=bins),
            Vint.name: Vint(model=model, acquire=acquire, bins=bins),
            Vdepth.name: Vdepth(model=model, acquire=acquire, bins=bins),
        }

        for input in inputs.values():
            input.train_on_shots = True  # 1D shots are CMPs.
            input.mute_dir = True
        for output in outputs.values():
            output.train_on_shots = True
            output.identify_direct = False

        return model, acquire, inputs, outputs


class Article1DDZMaxAll(Article1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsets = {
            dzmax: dataset(*args, **kwargs)
            for dzmax, dataset in Article1DDZMax.items()
            if dzmax != 'all'
        }
        self.trainsize = 600

    def tfdataset(self, phase='train', *args, **kwargs):
        if phase == "validate" and self.validatesize == 0:
            return
        super().tfdataset(phase, *args, **kwargs)
        self.subsets = {
            dip: subset.tfdataset(phase, *args, **kwargs)
            for dip, subset in self.subsets.items()
        }
        for subset in self.subsets.values():
            subset.tfdataset(phase, *args, **kwargs)
        self.files = {
            key: list(
                chain(
                    *(subset.files[key] for subset in self.subsets.values())
                )
            )
            for key in ['train', 'validate', 'test']
        }
        np.random.shuffle(self.files[self.phase])
        self.iter_examples = cycle(self.files[self.phase])
        self.on_epoch_end()
        return self


class Article2D(Article1D):
    def set_dataset(self):
        model, acquire, inputs, outputs = Article1D.set_dataset(self)

        self.trainsize = 200
        self.testsize = 10

        model.max_deform_freq = .06
        model.min_deform_freq = .0001
        model.amp_max = 8
        model.max_deform_nfreq = 40
        model.prob_deform_change = .7
        model.ddip_max = model.dip_max / 2

        acquire.singleshot = False

        for input in inputs.values():
            input.train_on_shots = False
        for output in outputs.values():
            output.train_on_shots = False

        return model, acquire, inputs, outputs


class Article2DDip(Article2D):
    def set_dataset(self):
        model, acquire, inputs, outputs = Article2D.set_dataset(self)

        self.trainsize = 200

        model.dh /= 2
        model.NX *= 2
        model.NZ *= 2
        acquire.dg *= 2
        acquire.ds *= 2
        acquire.gmin *= 2
        acquire.gmax *= 2

        acquire.dt /= 2
        acquire.NT *= 2
        acquire.resampling *= 2

        inputs = {
            ShotGather.name: ShotGatherCrop(model=model, acquire=acquire)
        }
        bins = self.params.decode_bins
        outputs = {
            Reftime.name: ReftimeCrop(model=model, acquire=acquire),
            Vrms.name: VrmsCrop(model=model, acquire=acquire, bins=bins),
            Vint.name: VintCrop(model=model, acquire=acquire, bins=bins),
            Vdepth.name: VdepthCrop(model=model, acquire=acquire, bins=bins),
        }
        for input in inputs.values():
            input.train_on_shots = False
            input.mute_dir = True
        for output in outputs.values():
            output.train_on_shots = False
            output.identify_direct = False

        return model, acquire, inputs, outputs


class DatasetPlaceholder(Article1D):
    @classmethod
    def construct(cls, name):
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        return cls

    def set_dataset(self, *args, **kwargs):
        model, acquire, inputs, outputs = Article1D.set_dataset(
            self, *args, **kwargs,
        )
        self.trainsize = 200
        self.testsize = 10
        return model, acquire, inputs, outputs

    def generate_dataset(self, *args, **kwargs):
        dataset = Article1DDZMax[1000](self.params)
        makedirs(self.datatest)
        for i in range(self.trainsize, self.trainsize+self.testsize):
            src = join(dataset.datatrain, f'example_{i}')
            dst = join(self.datatest, f'example_{i}')
            fcopy(src, dst)

    def get_example(self, *args, **kwargs):
        NS = 52

        *dicts, filename = Article1D.get_example(self, *args, **kwargs)
        for dict in dicts:
            for key, array in dict.items():
                axis = -1 if array.ndim == 2 else -2
                array = np.repeat(array, NS, axis=axis)
                dict[key] = array
        return (*dicts, filename)


Article1DDZMax = {
    **{
        dzmax: Article1D.construct(
            suffix=f'DZMax{dzmax}', dzmax=dzmax,
        )
        for dzmax in [250, 500, 1000, 2000]
    },
}
Article1DFreq = {
    peak_freq: Article1D.construct(
        suffix=f'Freq{peak_freq}', peak_freq=peak_freq,
    )
    for peak_freq in [10, 15, 25, 40]
}
Article2DDip = {
    # 0: Article1D.construct(suffix='DZMax1000'),
    **{
        dip_max: Article2DDip.construct(
            suffix=f'{dip_max}', dip_max=dip_max,
        )
        for dip_max in [10, 25, 40]
    },
}
Article2DFault = {
    0: DatasetPlaceholder.construct('Article2DFault0'),
    **{
        displ: Article2D.construct(
            suffix=f'Fault{displ}', fault_displ_min=displ, fault_prob=1,
        )
        for displ in [-200, -400]
    },
}


class InterpolateDips(Article2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsets = {
            dip: dataset(*args, **kwargs)
            for dip, dataset in Article2DDip.items()
            if dip != 0
        }
        self.current_epoch = -1

    @property
    def proportions(self):
        return [
            sum(dataset.name in f for f in self.files[self.phase])
            for dataset in self.subsets.values()
        ]

    def tfdataset(self, phase='train', *args, **kwargs):
        if phase == "validate" and self.validatesize == 0:
            return
        super().tfdataset(phase, *args, **kwargs)
        self.subsets = {
            dip: subset.tfdataset(phase, *args, **kwargs)
            for dip, subset in self.subsets.items()
        }
        for subset in self.subsets.values():
            subset.tfdataset(phase, *args, **kwargs)
        self.files = self.subsets[10].files
        np.random.shuffle(self.files[self.phase])
        self.iter_examples = cycle(self.files[self.phase])
        self.on_epoch_end()
        return self

    def on_epoch_end(self):
        self.current_epoch += 1
        for _ in range(5):
            if self.current_epoch >= 20:
                self.replace_file(25, 40)
            self.replace_file(10, 25)
        np.random.shuffle(self.files[self.phase])
        self.iter_examples = cycle(self.files[self.phase])

    def replace_file(self, source_dip, target_dip):
        source_dataset = self.subsets[source_dip]
        target_dataset = self.subsets[target_dip]
        replaceable = [
            f for f in self.files[self.phase] if source_dataset.name in f
        ]
        file = np.random.choice(replaceable)
        idx = self.files[self.phase].index(file)
        self.files[self.phase][idx] = file.replace(
            source_dataset.name, target_dataset.name,
        )


class MarineModel(MarineModel):
    def build_stratigraphy(self):
        self.thick0min = int(self.water_dmin/self.dh)
        self.thick0max = int(self.water_dmax/self.dh)

        vp = Property(
            name="vp", vmin=self.water_vmin, vmax=self.water_vmax, dzmax=0,
        )
        vs = Property(name="vs", vmin=0, vmax=0)
        rho = Property(name="rho", vmin=2000, vmax=2000)
        water = Lithology(name='water', properties=[vp, vs, rho])
        vp = Property(
            name="vp",
            vmin=self.vp_min,
            vmax=self.vp_max,
            texture=self.max_texture,
            trend_min=self.vp_trend_min,
            trend_max=self.vp_trend_max,
            dzmin=self.dzmin,
            dzmax=self.dzmax,
            filter_decrease=self.accept_decrease > 0,
        )
        roc = Lithology(name='roc', properties=[vp, vs, rho])
        if self.amp_max > 0 and self.max_deform_nfreq > 0:
            deform = Deformation(
                max_deform_freq=self.max_deform_freq,
                min_deform_freq=self.min_deform_freq,
                amp_max=self.amp_max,
                max_deform_nfreq=self.max_deform_nfreq,
                prob_deform_change=self.prob_deform_change,
            )
        else:
            deform = None
        waterseq = GeoSequence(
            lithologies=[water],
            ordered=False,
            thick_min=self.thick0min,
            thick_max=self.thick0max,
            nmin=1,
        )
        rocseq = GeoSequence(
            lithologies=[roc],
            ordered=False,
            deform=deform,
            accept_decrease=self.accept_decrease,
        )
        strati = Stratigraphy(sequences=[waterseq, rocseq])
        properties = strati.properties()

        return strati, properties


class Vrms(Vrms):
    def __init__(self, model, acquire, bins):
        super().__init__(model, acquire)
        self.bins = bins

    def plot(
        self, data, weights=None, axs=None, cmap='inferno', vmin=None,
        vmax=None, clip=1, ims=None, std_min=None, std_max=None,
    ):
        max_, std = data
        if weights is not None:
            weights = weights[..., 0, 0]

        return super().plot(max_, weights, axs, cmap, vmin, vmax, clip, ims)

    def postprocess(self, output):
        median, std = self.reduce(output)
        vmin, vmax = self.model.properties["vp"]
        median = median*(vmax-vmin) + vmin
        std = std * (vmax-vmin)
        return median, std

    def reduce(self, output):
        if output.ndim > 2 and output.shape[2] > 1:
            while output.ndim > 3:
                assert output.shape[-1] == 1
                output = output[..., 0]
            prob = output
            bins = np.linspace(0, 1, self.bins+1)
            bins = np.mean([bins[:-1], bins[1:]], axis=0)
            v = np.zeros_like(prob)
            v[:] = bins[None, None]
            median = weighted_median(v, weights=prob, axis=-1)
            mean = np.average(v, weights=prob, axis=-1)
            var = np.average((v-mean[..., None])**2, weights=prob, axis=-1)
            std = np.sqrt(var)
        else:
            median = output
            while median.ndim > 2:
                median = median[..., 0]
            std = np.zeros_like(median)
        return median, std


class Vint(Vrms, Vint):
    pass


class Vdepth(Vrms, Vdepth):
    pass


class ShotGather(ShotGather):
    def plot(
        self, data, weights=None, axs=None, cmap='Greys', vmin=0, vmax=None,
        clip=.08, ims=None,
    ):
        if data.shape[2] == 1 and weights is not None:
            weights = np.repeat(weights, data.shape[1], axis=1)
        return super().plot(data, weights, axs, cmap, vmin, vmax, clip, ims)


def weighted_median(array, weights, axis):
    weights /= np.sum(weights, axis=axis, keepdims=True)
    weights = np.cumsum(weights, axis=axis)
    weights = np.moveaxis(weights, axis, 0)
    len_axis, *source_shape = weights.shape
    weights = weights.reshape([len_axis, -1])
    median_idx = [np.searchsorted(w, .5) for w in weights.T]
    array = np.moveaxis(array, axis, 0)
    array = array.reshape([len_axis, -1]).T
    median = array[np.arange(len(array)), median_idx]
    median = median.reshape(source_shape)
    return median
