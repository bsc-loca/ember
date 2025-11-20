
import numpy as np
import matplotlib.pyplot as plt
from default_config import *
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
figure_size = (10,3)
xtitle = ''
ytitle = 'Opt speedup'
bbox = (-0.01, 0.78)
legend_ncol = 3
legend_loc = 3

xtitle_fontsize=20      # x axis title
ytitle_fontsize=20      # y axis title
xlabel_fontsize=16      # x axis labels
ylabel_fontsize=16      # y axis labels
xticks_rotation = 45


# From gem5 simulator
o3_cycles = [3596370, 3773389, 2143108, 3823321, 3602097, 2912381, 3136845, 2902392, 2516575, 48132732, 13527835, 8959858, 160584040]
o2_cycles = [4156513, 4280459, 2397315, 5024453, 5055357, 4071356, 4897272, 4514832, 3964262, 54901397, 15782474, 10579109, 191664822]
o1_cycles = [5930378, 5717256, 3337547, 8442575, 8640787, 6612184, 10922205, 10006250, 8650471, 83659272, 24703003, 16567285, 292210302]
o0_cycles = [23484296, 24527029, 14551703, 45421053, 43981604, 35705791, 66407009, 60137562, 53200396, 351368944, 113633814, 87806608, 1782482844]

o1_speedup = [o0c/o1c for o0c, o1c in zip(o0_cycles, o1_cycles)]
o2_speedup = [o0c/o2c - o1s for o0c, o2c, o1s in zip(o0_cycles, o2_cycles, o1_speedup)]
o3_speedup = [o0c/o3c - o1s - o2s for o0c, o3c, o1s, o2s in zip(o0_cycles, o3_cycles, o1_speedup, o2_speedup)]

experiments = [
    'RM1_L0',
    'RM1_L1',
    'RM1_L3',
    'RM2_L0',
    'RM2_L1',
    'RM2_L3',
    'RM3_L0',
    'RM3_L1',
    'RM3_L3',
    'wiki-Talk',
    'roadNet-CA',
    'com-Youtube',
    'web-Google',
    ]


# labels, use benchmark names
xticks = experiments

# column names
if auto_column_names:
    legend = ['emb-opt1','emb-opt2','emb-opt3']
    column_ids_data = range(1, len(legend)+1)

ind = np.arange(len(xticks))    # the x locations for the groups
barwidth = 1.0/(2)  # the width of the bars
spacing = barwidth / (2)

# create a new figure and axes instance
fig = plt.figure(figsize=figure_size) # figure size specified in config
ax = fig.add_subplot(111)

ylim = (0, 25)

# Set ylim and xlim
if ylim:
    ax.set_ylim(*ylim)
    ax.set_xlim(right=len(ind))

if yticks != None:
    ax.set_yticks(yticks)

ax.set_yscale('linear')
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_xscale(xscale)

# Generate all bars

import seaborn as sns
colors = sns.color_palette(palette='Blues')

rects = []
left_empty = barwidth
rects.append(ax.bar(x=left_empty+ind, height=o1_speedup, width=barwidth, alpha=1,
                    color=colors[5], ecolor='black', edgecolor='black', linewidth=0.5))
rects.append(ax.bar(x=left_empty+ind, bottom=o1_speedup, height=o2_speedup, width=barwidth, alpha=1,
                    color=colors[3], ecolor='black', edgecolor='black', linewidth=0.5))
rects.append(ax.bar(x=left_empty+ind, bottom=[a + b for a,b in zip(o1_speedup, o2_speedup)], height=o3_speedup,
                    width=barwidth, alpha=1, color=colors[1], ecolor='black', edgecolor='black', linewidth=0.5))

# general formating
set_titles(ax, title, xtitle, ytitle, title_fontsize,
           xtitle_fontsize, ytitle_fontsize, ylabel_fontsize)


# xticks possition and labels
ax.set_xticks(ind + left_empty + 0.1)
ax.set_xticklabels(xticks, fontsize=xlabel_fontsize, rotation=xticks_rotation, ha="right")
ax.tick_params(axis='x',length=0)
plt.gcf().subplots_adjust(bottom=0.2)

# Set custom ticks on the y-axis using FuncFormatter
from matplotlib.ticker import FuncFormatter
def x_ticks(y, pos):
    return f'{int(y)}x'
ax.yaxis.set_major_formatter(FuncFormatter(x_ticks))

ax.axvline(9, color='gray', linestyle='--')

# legend
if do_legend:
    leg = ax.legend([a[0] for a in rects],
                    legend,
                    loc=0,
                    ncol=1,
                    #frameon=True,
                    #borderaxespad=1.,
                    #bbox_to_anchor=bbox,
                    handlelength=1,         # Shorter lines in legend
                    handletextpad=0.3,      # Reduce space between legend marker and text
                    labelspacing=0.2,       # Reduce vertical space between labels
                    fancybox=True,
                    reverse=True)
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
plt.savefig("fig_13.pdf",bbox_inches='tight')