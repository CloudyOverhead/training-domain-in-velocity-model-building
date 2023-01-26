# -*- coding: utf-8 -*-

"""Custom Matplotlib plot format."""

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator
import proplot as pplt
import numpy as np


class FuncScale(pplt.FuncScale):
    def __init__(self, *, a=1, b=0, decimals=None):
        def major_formatter(x, _):
            y = a*x + b
            if decimals is not None:
                y = np.around(y, decimals=decimals)
                if decimals <= 0:
                    y = y.astype(int)
            return y

        super().__init__(
            transform=lambda x: x,
            major_locator=FuncLocator(a=a, b=b),
            major_formatter=major_formatter,
        )


class FuncLocator(AutoLocator):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def tick_values(self, vmin, vmax):
        vmin += self.b / self.a
        vmax += self.b / self.a
        ticks = super().tick_values(vmin, vmax)
        ticks *= self.a
        ticks -= self.b
        ticks /= self.a
        return ticks


pplt.FuncScale = FuncScale


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
plt.rcParams['text.usetex'] = True
plt.rcParams.update(plt.rcParamsDefault)

plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.style'] = "normal"
plt.rcParams['font.variant'] = "normal"
plt.rcParams['font.weight'] = "medium"
plt.rcParams['font.stretch'] = "normal"
plt.rcParams['font.size'] = 8
# plt.rcParams['font.serif'] = (
#     DejaVu Serif, Bitstream Vera Serif, New Century Schoolbook,
#     Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L,
#     Times New Roman, Times, Palatino, Charter, serif,
# )
# plt.rcParams['font.sans-serif'] = (
#     DejaVu Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid,
#     Arial, Helvetica, Avant Garde, sans-serif,
# )
# plt.rcParams['font.cursive'] = (
#     Apple Chancery, Textile, Zapf Chancery, Sand, Script MT, Felipa, cursive,
# )
# plt.rcParams['font.fantasy'] = (
#     Comic Sans MS, Chicago, Charcoal, Impact, Western, Humor Sans, xkcd,
#     fantasy,
# )
# plt.rcParams['font.monospace'] = (
#     DejaVu Sans Mono, Bitstream Vera Sans Mono, Andale Mono, Nimbus Mono L,
#     Courier New, Courier, Fixed, Terminal, monospace,
# )

# plt.rcParams['axes.facecolor'] = "white"  # axes background color
# plt.rcParams['axes.edgecolor'] = "black"  # axes edge color
# plt.rcParams['axes.linewidth'] = 0.8  # edge linewidth
# plt.rcParams['axes.grid'] = False  # display grid or not
# plt.rcParams['axes.titlesize'] = 8  # fontsize of the axes title
# plt.rcParams['axes.titlepad'] = 4  # pad between axes and title in points
# plt.rcParams['axes.labelsize'] = 8  # fontsize of the x any y labels
# plt.rcParams['axes.labelpad'] = 4.0  # space between label and axis
# plt.rcParams['axes.labelweight'] = "normal"  # weight of the x and y labels
# plt.rcParams['axes.labelcolor'] = "black"  # label color
# plt.rcParams['axes.axisbelow'] = 'line'  # draw axis gridlines and ticks below

# XTICKS
# plt.rcParams['xtick.top'] = True  # draw ticks on the top side
# plt.rcParams['xtick.bottom'] = True  # draw ticks on the bottom side
plt.rcParams['xtick.minor.visible'] = True
# plt.rcParams['xtick.major.size'] = 3.5  # major tick size in points
# plt.rcParams['xtick.minor.size'] = 2  # minor tick size in points
# plt.rcParams['xtick.major.width'] = 0.8  # major tick width in points
# plt.rcParams['xtick.minor.width'] = 0.6  # minor tick width in points
# plt.rcParams['xtick.major.pad'] = 3.5  # distance to major tick label in points
# plt.rcParams['xtick.minor.pad'] = 3.4  # distance to minor tick label in points
# plt.rcParams['xtick.color'] = "k"  # color of the tick labels
# plt.rcParams['xtick.labelsize'] = 16  # fontsize of the tick labels
# plt.rcParams['xtick.direction'] = 'in'  # direction: in, out, or inout
# plt.rcParams['xtick.major.top'] = True  # draw x axis top major ticks
# plt.rcParams['xtick.major.bottom'] = True  # draw x axis bottom major ticks
# plt.rcParams['xtick.minor.top'] = True  # draw x axis top minor ticks
# plt.rcParams['xtick.minor.bottom'] = True  # draw x axis bottom minor ticks

# YTICKS
# plt.rcParams['ytick.left'] = True  # draw ticks on the left side
# plt.rcParams['ytick.right'] = True  # draw ticks on the right side
plt.rcParams['ytick.minor.visible'] = True
# plt.rcParams['ytick.major.size'] = 3.5  # major tick size in points
# plt.rcParams['ytick.minor.size'] = 2  # minor tick size in points
# plt.rcParams['ytick.major.width'] = 0.8  # major tick width in points
# plt.rcParams['ytick.minor.width'] = 0.6  # minor tick width in points
# plt.rcParams['ytick.major.pad'] = 3.5  # distance to major tick label in points
# plt.rcParams['ytick.minor.pad'] = 3.4  # distance to minor tick label in points
# plt.rcParams['ytick.color'] = "k"  # color of the tick labels
# plt.rcParams['ytick.labelsize'] = 8  # fontsize of the tick labels
# plt.rcParams['ytick.direction'] = "in"  # direction: in, out, or inout
# plt.rcParams['ytick.minor.visible'] = False  # visibility of minor y-axis ticks
# plt.rcParams['ytick.major.left'] = True  # draw y axis left major ticks
# plt.rcParams['ytick.major.right'] = True  # draw y axis right major ticks
# plt.rcParams['ytick.minor.left'] = True  # draw y axis left minor ticks
# plt.rcParams['ytick.minor.right'] = True  # draw y axis right minor ticks

# GRIDS
# plt.rcParams['grid.linestyle'] = "-"  # solid
# plt.rcParams['grid.linewidth'] = 0.8  # in points
# plt.rcParams['grid.alpha'] = 1.0  # transparency, between 0.0 and 1.0

# Legend
# plt.rcParams['legend.loc'] = "best"
# plt.rcParams['legend.frameon'] = True  # draw the legend on a background patch
# plt.rcParams['legend.framealpha'] = 0.8  # legend patch transparency
plt.rcParams['legend.facecolor'] = [1, 1, 1]  # inherit `axes.facecolor`; color
# plt.rcParams['legend.edgecolor'] = "k"  # background patch boundary color
# plt.rcParams['legend.fancybox'] = True  # use a rounded box for the
# plt.rcParams['legend.shadow'] = False  # give background a shadow effect
# plt.rcParams['legend.numpoints'] = 1  # number of marker points in legend line
# plt.rcParams['legend.scatterpoints'] = 1  # number of scatter points
# plt.rcParams['legend.markerscale'] = 1.0  # relative size of markers
# plt.rcParams['legend.fontsize'] = 8
# plt.rcParams['legend.borderpad'] = 0.4  # border whitespace
# plt.rcParams['legend.labelspacing'] = 0.5  # vertical space between entries
# plt.rcParams['legend.handlelength'] = 2.0  # the length of the legend lines
# plt.rcParams['legend.handleheight'] = 0.7  # the height of the legend handle
# plt.rcParams['legend.handletextpad'] = 0.8  # the space between the line text
# plt.rcParams['legend.borderaxespad'] = 0.5  # the border between axes and edge
# plt.rcParams['legend.columnspacing'] = 2.0  # column separation

# FIGURE
# plt.rcParams['figure.titlesize'] = "large"  # size of `Figure.suptitle()`
# plt.rcParams['figure.titleweight'] = "normal"  # weight of the figure title
plt.rcParams['figure.figsize'] = [4.33, 3]  # figure size in inches
plt.rcParams['figure.dpi'] = 1000  # figure dots per inch
# plt.rcParams['figure.facecolor'] = "white"  # figure facecolor
# plt.rcParams['figure.edgecolor'] = "white"  # figure edgecolor
# plt.rcParams['figure.autolayout'] = False  # automatically adjust subplot.

pplt.rc.update(
    grid=True,
    gridalpha=.2,
    gridcolor='k',
    gridminor=True,
    gridminorcolor='k',
    gridminoralpha=.2,
    gridminorlinestyle=':',
)
