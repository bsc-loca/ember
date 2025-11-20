
import numpy as np
import matplotlib.pyplot as plt

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
figure_size = (7,2.5)
xtitle = 'Compute throughput'
ytitle = 'Access throughput'
ylim = (0, 60)
xlim = (0, 80)
bbox = (-0.01, 0.45)
legend_ncol = 3
legend_loc = 3

xtitle_fontsize = 14      # x axis title
ytitle_fontsize = 14      # y axis title
xlabel_fontsize = 12      # x axis labels
ylabel_fontsize = 12      # y axis labels
xticks_rotation = 60
xticks_rotation = 90

color =[
    [0.43137255, 0.08627451, 0.36470588, 1.        ],
    [0.60084583, 0.35860054, 0.55404844, 1.        ],
    [0.77254902, 0.6345098,  0.74588235, 1.        ],

    [0.12941176, 0.22352941, 0.43921569, 1.        ],
    [0.3888812,  0.4549481,  0.6063514,  1.        ],
    [0.74053057, 0.76858131, 0.83286428, 1.        ],

    [0.08627451, 0.43137255, 0.15294118, 1.        ],
    [0.35860054, 0.60084583, 0.40539792, 1.        ],
    [0.6345098,  0.77254902, 0.66117647, 1.        ],
]

color2 =[
    [0.73137255, 0.08627451, 0.6470588,  0.75        ],
    [0.60084583, 0.35860054, 0.55404844, 1.        ],
    [0.77254902, 0.6345098,  0.74588235, 1.        ],

    [0.12941176, 0.22352941, 0.43921569, 1.        ],
    [0.3888812,  0.4549481,  0.6063514,  1.        ],
    [0.74053057, 0.76858131, 0.83286428, 1.        ],

    [0.08627451, 0.43137255, 0.15294118, 1.        ],
    [0.35860054, 0.60084583, 0.40539792, 1.        ],
    [0.6345098,  0.77254902, 0.66117647, 1.        ],
]

lineargs={
    "linewidth"     : 3,
    "markersize"    : 12
}

linecolors = (color[0],color[1],color[2],color[3],color[4],color[5],color[6],color[7],color[8])
linecolors2 = (color2[0],color2[1],color2[2],color2[3],color2[4],color2[5],color2[6],color2[7],color2[8])

import seaborn as sns
c1 = sns.color_palette(palette='Greys')
c2 = sns.color_palette(palette='Greens')
c3 = sns.color_palette(palette='Blues')
i1 = 4
i2 = 3
i3 = 2
linecolors3 = (c1[i1], c1[i2], c1[i3], c2[i1], c2[i2], c2[i3], c3[i1], c3[i2], c3[i3])

marker_patterns = ('o','^','s','*','P','v','o','^','s','*','P','o','^','s','*','.','o','^','s','*','.','o','^','s','*','.',)
line_styles = ('-','-','--','--','--','--','--','-','--','-','--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--')


header = ['RM1_L0', 'RM1_L1', 'RM1_L2', 'RM2_L0', 'RM2_L1', 'RM2_L2', 'RM3_L0', 'RM3_L1', 'RM3_L2']
header2 = ['vec', 'buf', 'idx',]

# Fom gem5 simulator (values are percentages)
x1 = [6.02, 14.44, 17.50, 18.05,]
x2 = [5.99, 14.38, 17.42, 17.97,]
x3 = [5.65, 13.56, 16.44, 16.95,]
x4 = [13.03, 31.27, 37.91, 39.09,]
x5 = [14.08, 33.80, 40.97, 42.25,]
x6 = [11.87, 28.49, 34.53, 35.61,]
x7 = [23.41, 56.19, 68.11, 70.23,]
x8 = [22.95, 55.08, 66.76, 68.85,]
x9 = [23.32, 55.96, 67.83, 69.95,]

data1 = [2.10, 8.31, 11.85, 13.78,]
data2 = [2.12, 9.14, 12.11, 13.73,]
data3 = [2.09, 9.13, 12.71, 14.23,]
data4 = [2.22, 11.96, 20.11, 26.39,]
data5 = [2.22, 11.30, 19.30, 27.07,]
data6 = [2.22, 11.98, 19.29, 27.17,]
data7 = [2.30, 13.98, 31.17, 48.62,]
data8 = [2.31, 13.91, 30.80, 47.98,]
data9 = [2.32, 14.23, 31.07, 48.93,]

x = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
data = [data1, data2, data3, data4, data5, data6, data7, data8, data9,]

# create a new figure and axes instance
fig = plt.figure(figsize=figure_size) # figure size specified in config
ax = fig.add_subplot(111)

ax.set_xscale('linear')
ax.set_yscale('linear')

# Set ylim and xlim
ax.set_ylim(*ylim)
ax.set_xlim(*xlim)

ax.tick_params(axis='both', which='major', pad=5)

# Plot all lines
mylines = []
mypoints = []
texts = []
for i,d in enumerate(header):
    mylines.append(ax.quiver(np.asarray(x[i][:-1]), np.asarray(data[i][:-1]),
                             np.asarray(x[i][1:])-np.asarray(x[i][:-1]),
                             np.asarray(data[i][1:])-np.asarray(data[i][:-1]),
                             scale_units='xy', angles='xy', scale=1,
                             color=linecolors3[i],
                             ))

# general formating
set_titles(ax, title, xtitle, ytitle, 0,
           xtitle_fontsize, ytitle_fontsize, ylabel_fontsize)

# Graph shrinking if desired, no shrinking by default
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])

ax.text(x[6][0]+2,data[6][0]+1,'emb-opt0', va="center", fontsize=12)
ax.text(x[6][1]+1,data[6][1]-2,'emb-opt1', va="center", fontsize=12)
ax.text(x[6][2],data[6][2]-3,'emb-opt2', va="center", fontsize=12)
ax.text(x[6][3]-2,data[6][3]+2,'emb-opt3', va="center", fontsize=12)

# legend
leg = ax.legend([a for a in mylines],
                header+header2,
                loc="upper left",
                ncol=legend_ncol,
                frameon=True,
                borderaxespad=1.,
                labelspacing=0.1,       # Reduce vertical space between labels
                #bbox_to_anchor=bbox,
                fancybox=True,
                #prop={'size':10}, # smaller font size
                )

ax.plot([0,100],[0,100])

#ax.set_axisbelow(True)
ax.yaxis.grid(color='0.5', linestyle='--', linewidth=0.3)

from matplotlib.ticker import FuncFormatter
def perc_ticks(y, pos):
    return f'{int(y)}%'
# Set custom ticks on the y-axis using FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(perc_ticks))
ax.yaxis.set_major_formatter(FuncFormatter(perc_ticks))

fig.tight_layout()
fig.savefig("fig_14.pdf",bbox_inches='tight')