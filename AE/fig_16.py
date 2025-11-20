import numpy as np
import matplotlib.pyplot as plt
from default_config import *
from matplotlib.ticker import ScalarFormatter

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
figure_size = (7,3.5)
xtitle = ''
ytitle = 'Ember vs hand-optimized code'
bbox = (-0.017, 0.965)
legend_ncol = 3
legend_loc = 3

xtitle_fontsize=20      # x axis title
ytitle_fontsize=12      # y axis title
xlabel_fontsize=12      # x axis labels
ylabel_fontsize=12      # y axis labels
xticks_rotation = 10

# From gem5 simulator (cycles_handwritten, cycles_ember)
results = {
    'wiki-Talk' : (160584040, 163029482),
    'roadNet-CA' : (13527835, 13733843),
    'com-Youtube' : (48132732, 48618921),
    'web-Google' : (8959858, 9004882),

    'arxiv' : (24914014, 26170183),
    'mag' : (30750063, 31441782),
    'products' : (31461428, 32103498),
    'proteins' : (21083914, 21189863),

    'bioKg' : (821426, 821426),
    'wikiKg2' : (5132043, 5132043),

    'RM1_L0' : (3596370, 3769780),
    'RM1_L1' : (3773389, 3951193),
    'RM1_L2' : (2143108, 2239402),
    'RM2_L0' : (3823321, 3961991),
    'RM2_L1' : (3602097, 3728879),
    'RM2_L2' : (2912381, 3008658),
    'RM3_L0' : (3136845, 3210691),
    'RM3_L1' : (2902392, 2967681),
    'RM3_L2' : (2516575, 2570557),

    '1_emb/blk' : (10463829, 10463829),
    '2_emb/blk' : (21485736, 21485736),
    '4_emb/blk' : (48372615, 48372615),
    '8_emb/blk' : (82745210, 82745210),
}

speedups = list([100.0 * t1 / t2 for t1, t2 in results.values()])

# labels, use benchmark names
xticks = list(results.keys())

ind = np.arange(len(xticks))    # the x locations for the groups
barwidth = 1.0/(2)  # the width of the bars
spacing = barwidth / (2)

# create a new figure and axes instance
fig = plt.figure(figsize=figure_size) # figure size specified in config
ax = fig.add_subplot(111)

ylim = (60, 109)

# Set ylim and xlim
if ylim:
    ax.set_ylim(*ylim)
    ax.set_xlim(left=-barwidth,right=len(ind)-0.25)

if yticks != None:
    ax.set_yticks(yticks)

ax.set_yscale('linear')
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_xscale(xscale)


import seaborn as sns
colors = sns.color_palette(palette='tab20b')

rects = []
left_empty = barwidth/4.0
rects.append(ax.bar(x=left_empty+ind, height=speedups, width=barwidth, alpha=1,
                    color=colors[0], ecolor='black', edgecolor='black', linewidth=0.5))

# general formating
set_titles(ax, title, xtitle, ytitle, title_fontsize,
           xtitle_fontsize, ytitle_fontsize, ylabel_fontsize)

# xticks possition and labels
ax.set_xticks(ind )
ax.set_xticklabels(xticks, fontsize=xlabel_fontsize, rotation=xticks_rotation, rotation_mode='anchor', transform_rotates_text=True, verticalalignment="top", horizontalalignment="right")
ax.tick_params(axis='x',length=0)
plt.gcf().subplots_adjust(bottom=0.2)

# Set custom ticks on the y-axis using FuncFormatter
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{int(y)}\%'))

vlines = [3.625, 7.625, 9.625, 18.625]

# Draw horizontal lines
for line in vlines:
    ax.axvline(line, color='gray', linestyle='--')

hlines = []

for line in hlines:
    ax.axhline(line, color='k',linestyle='-')

xx = 0.10#-0.25
yy = 105
sz = 1.3*numbers_fontsize
ax.text(x=xx+1.5, y=yy, s='MP', ha='center', va='center', fontweight='bold', fontsize=sz)
ax.text(x=xx+5.55, y=yy, s='SpMM(GNN)', ha='center', va='center', fontweight='bold', fontsize=sz)
ax.text(x=xx+8.5, y=yy, s='KG', ha='center', va='center', fontweight='bold', fontsize=sz)
ax.text(x=xx+14, y=yy, s='EB(DLRM)', ha='center', va='center', fontweight='bold', fontsize=sz)
ax.text(x=xx+19, y=yy, s='SpAttn(LLM)', ha='left', va='center', fontweight='bold', fontsize=sz/1.3*1.09)

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
plt.savefig("fig_16.pdf", bbox_inches='tight', format="pdf")
