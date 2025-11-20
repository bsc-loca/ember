
import os
import sys
import string
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from functools import reduce
from math import log, atan2, degrees
from matplotlib.colors import colorConverter
from collections import OrderedDict
from default_config import *
from pprint import pprint
#from adjustText import adjust_text
from matplotlib.ticker import AutoLocator, ScalarFormatter

def set_titles(ax, title, xtitle, ytitle, title_fontsize,
               xtitle_fontsize, ytitle_fontsize, ylabel_fontsize):
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xtitle, fontsize=xtitle_fontsize)
    ax.set_ylabel(ytitle, fontsize=ytitle_fontsize)
    for item in ax.get_yticklabels():
        item.set_fontsize(ylabel_fontsize)
    for item in ax.get_xticklabels():
        item.set_fontsize(xlabel_fontsize)

title = ""
figure_size = (10,2.3)
xtitle = ''
ytitle = 'L3 APKE'
bbox = (0.55, 0.765)
legend_ncol = 3
legend_loc = 3

xtitle_fontsize=20      # x axis title
ytitle_fontsize=16      # y axis title
xlabel_fontsize=16      # x axis labels
ylabel_fontsize=16      # y axis labels
xticks_rotation = 60
xticks_rotation = 90

# From gem5 simulator
non_temporal_apke = [21.03, 21.02, 7.74, 7.92, 3.66, 3.38, 1.41, 1.40,]
temporal_apke = [62.99, 20.73, 31.55, 10.35, 16.02, 5.18, 9.99, 2.61,]
experiments = ['L3_1', 'L2_1', 'L3_2', 'L2_2', 'L3_4', 'L2_4', 'L3_8', 'L2_8',]


# labels, use benchmark names
xticks = experiments

# column names
if auto_column_names:
    legend = ['Non-temporal accesses','Temporal accesses']
    column_ids_data = range(1, len(legend)+1)

ind = np.arange(len(xticks))/1.5    # the x locations for the groups
ind = ind + [0, 0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9]
barwidth = 1.0/(2)  # the width of the bars

# create a new figure and axes instance
fig = plt.figure(figsize=figure_size) # figure size specified in config
ax = fig.add_subplot(111)

ylim = (0, 100)

xlim = (0, 6 + barwidth)

# Set ylim and xlim
if ylim:
    ax.set_ylim(*ylim)

ax.set_xlim(*xlim)

ax.set_yscale('linear')
ax.yaxis.set_major_formatter(ScalarFormatter())
#ax.set_xscale(xscale)

# Generate all bars
color1 = (0.59375,   0.6484375 , 0.77734375, 1.0)
color2 = (0.37109375,0.50390625, 0.78515625, 1.0)
color3 = (0.18359375,0.31640625, 0.62109375, 1.0)
color4 = (0.12109375,0.22265625, 0.4375,     1.0)
color5 = (0.0859375, 0.15625   , 0.29296875, 1.0)

colormapping = [color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2,color2]

rects = []
left_empty = barwidth
rects.append(ax.bar(x=left_empty+ind, height=non_temporal_apke, width=barwidth, alpha=1,
                    color=color4, ecolor='black', edgecolor='black', linewidth=0.5))
rects.append(ax.bar(x=left_empty+ind, bottom=non_temporal_apke, height=temporal_apke, width=barwidth, alpha=1,
                    color=color1, ecolor='black', edgecolor='black', linewidth=0.5))

# general formating
set_titles(ax, title, xtitle, ytitle, title_fontsize,
           xtitle_fontsize, ytitle_fontsize, ylabel_fontsize)


# xticks possition and labels
ax.set_xticks(ind + left_empty + 0.1)
ax.set_xticklabels(xticks, fontsize=xlabel_fontsize, rotation=45, ha="right")
ax.tick_params(axis='x',length=0)
plt.gcf().subplots_adjust(bottom=0.2)

# Set custom ticks on the y-axis using FuncFormatter
from matplotlib.ticker import FuncFormatter
def x_ticks(y, pos):
    return f'{int(y)}'
ax.yaxis.set_major_formatter(FuncFormatter(x_ticks))

# legend
if do_legend:
    leg = ax.legend([a[0] for a in rects],
                    legend,
                    loc="upper right",
                    ncol=1,
                    #frameon=True,
                    #borderaxespad=1.,
                    #bbox_to_anchor=bbox,
                    fancybox=True,
                    )
    for t in leg.get_texts():
        t.set_fontsize(legend_fontsize)    # the legend text fontsize
else:
    leg = ax.legend([], frameon=False)
# Draw horizontal lines
for line in hlines:
    ax.axhline(**line)
# Draw text labels
for text in text_labels:
    ax.text(**text)
# Graph shrinking if desired, no shrinking by default
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * shrink_width_factor, box.height * shrink_height_factor])
ax.set_axisbelow(True)
if line_split:
    ax2.set_axisbelow(True)
plt.gca().yaxis.grid(color='0.5', linestyle='--', linewidth=0.3)
plt.tight_layout()
plt.savefig("fig_15.pdf",bbox_inches='tight')