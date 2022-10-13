from os import listdir, remove
from os.path import join, exists, split
import re

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from h5py import File
from inflection import underscore


class regex_dict(dict):
    def __getitem__(self, key):
        if '*' in key:
            key = key.replace('*', '.*')
            matches = list(filter(re.compile(key).match, self.keys()))
            assert len(matches) == 1
            key = matches[0]
        return super().__getitem__(key)


class Catalog(list):
    dir, _ = split(__file__)
    dir = join(dir, 'figures')

    def __getitem__(self, idx):
        if isinstance(idx, str):
            try:
                idx = self.filenames.index(idx)
            except ValueError:
                raise ValueError(f"Figure `{idx}` is not registered.")
        return super().__getitem__(idx)

    @property
    def filenames(self):
        return [figure.filename.split('.')[0] for figure in self]

    def register(self, figure):
        if type(figure) is type:
            figure = figure()
        self.append(figure)

    def draw_all(self, gpus, show=True):
        for figure in self:
            figure.generate(gpus)
            figure.save(show=show)

    def regenerate(self, idx, gpus):
        figure = self[idx]
        metadata = figure.Metadata(gpus)
        metadata.generate(gpus)

    def regenerate_all(self, gpus):
        for i in range(len(self)):
            self.regenerate(i, gpus)

    def clear_all(self):
        for filename in listdir(self.dir):
            extension = filename.split('.')[-1]
            if extension in ['pdf', 'meta']:
                remove(join(self.dir, filename))


class Metadata(File):
    @property
    def filename(self):
        filename = underscore(type(self).__name__)
        filename = filename.strip('_')
        return filename + '.meta'

    @property
    def filepath(self):
        return join(Catalog.dir, 'metadata', self.filename)

    def __init__(self, gpus, *args, **kwargs):
        is_not_generated = not exists(self.filepath)
        super().__init__(self.filepath, 'a', *args, **kwargs)
        if is_not_generated:
            self.generate(gpus)

    def generate(self, gpus):
        raise NotImplementedError

    def __setitem__(self, key, value):
        try:
            del self[key]
        except KeyError:
            pass
        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        try:
            return value[:]
        except ValueError:
            return value[()]
        except AttributeError:
            return {item: self[key+'/'+item] for item in value.keys()}


class CompoundMetadata(regex_dict):
    @classmethod
    def combine(cls, *others):
        class cls(cls):
            pass
        cls._children = regex_dict()
        for child in others:
            if child.__name__ == cls.__name__:  # If is subclass.
                cls._children.update(child._children)
            else:
                cls._children[child.__name__] = child
        return cls

    def __init__(self, gpus):
        super().__init__()
        for key, child in self._children.items():
            self[key] = child(gpus)

    def __getitem__(self, key):
        name, *key = key.split('/')
        child = super().__getitem__(name)
        if key:
            key = '/'.join(key)
            return child[key]
        else:
            return child

    def __enter__(self):
        for child in self.values():
            child.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        for child in self.values():
            child.__exit__(*args, **kwargs)

    def generate(self, gpus):
        for child in self.values():
            child.generate(gpus)


class Figure(Figure):
    Metadata = Metadata

    @property
    def filename(self):
        return underscore(type(self).__name__) + '.pdf'
        # return f"Figure_{catalog.index(self)+1}.pdf"

    @property
    def filepath(self):
        return join(Catalog.dir, self.filename)

    def generate(self, gpus):
        with self.Metadata(gpus) as data:
            self.plot(data)

    def save(self, show=True):
        plt.savefig(self.filepath, transparent=True)
        if show:
            plt.show()
        else:
            plt.clf()
        plt.close()

    def plot(self, data):
        raise NotImplementedError


catalog = Catalog()
