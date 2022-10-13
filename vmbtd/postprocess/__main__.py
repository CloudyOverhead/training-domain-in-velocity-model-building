from argparse import ArgumentParser

from GeoFlow.__main__ import int_or_list

from vmbrc.postprocess.catalog import catalog


parser = ArgumentParser()
parser.add_argument('--list', '-l', action='store_true')
parser.add_argument('--metadata', '-m', action='store_true')
parser.add_argument('--figure', '-f', type=str)
parser.add_argument('--show', '-s', action='store_true')
parser.add_argument('--gpus', type=int_or_list)
args = parser.parse_args()

if args.list:
    print(catalog.filenames)
else:
    if args.metadata:
        if args.figure:
            catalog.regenerate(args.figure, args.gpus)
        else:
            catalog.regenerate_all(args.gpus)

    if args.figure:
        figure = catalog[args.figure]
        figure.generate(args.gpus)
        figure.save(show=args.show)
    else:
        catalog.draw_all(args.gpus, show=args.show)
