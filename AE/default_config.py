#!/usr/bin/env python

import matplotlib as mp
from matplotlib.colors import colorConverter

##################
### ALL charts ###
##################
chart_type = "barchart"

# name of the local configuration files
config_fname = "local.config.py"

# figure size (x,y)
figure_size = (5,4)

# ID of the column that holds the xtick (horizontal) labels
xticks_id = 0

# Rotation for xticks labels
xticks_rotation = 'horizontal'

# Column names
auto_column_names = True                        # use column names from header row of CSV file
column_names = ["First", "Second", "Third"]     # if auto_column_names is False specify column names here

# ID of columns holding data and error data (for error bars)
column_ids_data = [1, 2, 3]
column_ids_err = [] # not implemented

# titles and font sizes for figure, x and y axis
title = "from-filename"
xtitle='X-Axis Title'
ytitle='Y-Axis Title'
ytitle2='Y-Axis2 Title'
title_fontsize=20       # figure title
xtitle_fontsize=16      # x axis title
ytitle_fontsize=17      # y axis title
xlabel_fontsize=15.5    # x axis labels
ylabel_fontsize=16      # y axis labels
legend_fontsize=18      # legend
text_fontsize=14.5      # x axis cluster labels
numbers_fontsize=10     # for numbers above ylim

# Split for secondary axis
line_split = None

# y axes limit
ylim = None
ylim2 = None
num_yticks = None
num_xticks = None
yticks = None
xticks = None

# set axis scales
yscale = 'linear'
xscale = 'linear'
yscale_base = 1

# legend
bbox=(0, 0, 1, 1)       # legend located at the figure corner
legend_ncol = 3         # number of columns for the legend (1 means vertical)
legend_loc = 'best'
do_legend = True
# 'best'         : 0, (only implemented for axis legends)
# 'upper right'  : 1,
# 'upper left'   : 2,
# 'lower left'   : 3,
# 'lower right'  : 4,
# 'right'        : 5,
# 'center left'  : 6,
# 'center right' : 7,
# 'lower center' : 8,
# 'upper center' : 9,
# 'center'       : 10,

# add arithmetic and geometric means
# beaware turning both on won't work as expected
do_add_average = False
do_add_geomean = False

# labels for bars that overflow ylim
label_enable='no'                 # enable labels - valid values 'no', 'always', 'ylim'
label_angle_rotation=0          # rotation angle
label_y_space=0                 # distance after ylim

# relative possition to marker for text labels
xytext_tomarker = (10, -10)

# Allows modification of the figure bounding box in case some elements
# like the legend get cut off of the figure.
shrink_width_factor=1.
shrink_height_factor=1.

# hatching and marking patterns
hatch_patterns = ( ' ', 'oo', '', '\\\\', 'x', '++', '//', 'O', '..', 'O', '-')
marker_patterns = ( 'o' , 's' , 'd' , '<' , 'h' , '^' , 'p' , 'D' , 'H' ,
 '_' , '>' , 'v' , 'x' , ',', '+' , '*' , ',' , '.' , '1' , '2' , '3' , '4' , )
marker_sizes = range(8,30)

# color palette
### http://www.colourlovers.com/palette/944213/forever_lost?widths=0
color_palette_1 = "#5D4157 #838689 #A8CABA #CAD7B2 #EBE3AA"
### http://www.colourlovers.com/palette/1542449/fighting_the_9-5.?widths=0
color_palette_2 = "#262626 #475959 #689493 #9DC4C4 #EBDDC7"
### http://www.colourlovers.com/palette/580974/Adrift_in_Dreams
color_palette_3 = "#CFF09E #A8DBA8 #79BD9A #3B8686 #0B486B"

# bw palette
bw_palette_1 = "#DDDDDD #999999 #555555 #111111"
bw_palette_2 = "#DDDDDD #BBBBBB #999999 #777777 #555555 #333333"
bw_palette_3 = "#FFFFFF #DDDDDD #BBBBBB #999999 #777777 #555555 #333333 #111111"

# default is black and white
colors = [colorConverter.to_rgb(a) for a in (bw_palette_2).split()]

# use palette_blue2+palette_blue15 for colored
#colors = [colorConverter.to_rgb(a) for a in (palette_blue2 + palette_blue15).split()]

# extensions allows for raw data files
EXTENSIONS=['.csv']

# paper formating - allow only type 1 core fonts
mp.rcParams['ps.useafm'] = True
mp.rcParams['pdf.use14corefonts'] = True
mp.rcParams['text.usetex'] = True #Let TeX do the typsetting
#mp.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
mp.rcParams['font.family'] = 'sans-serif' # ... for regular text
mp.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here

# Draw horizontal lines using axhline(), accepts Line2D kwargs, see:
# http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D
hlines = [ # Add one dictionary per line, example:
        # { "y" : 1, "linewidth" : 1, "linestyle" : 'dashed'}
        ]

vlines = [ # Add one dictionary per line, example:
        # { "x" : 1, "linewidth" : 1, "linestyle" : 'dashed'}
        ]

# Draw text labels using text(), accepts Text kwargs, see:
# http://matplotlib.org/api/text_api.html#matplotlib.text.Text
text_labels = [ # Add one dictionary per text label, example:
        # { "x" : 0, "y" : 100, "s" : '1 day', "size" : 'x-large', "weight" : 'bold' }
        ]

points = [ # Add one dictionary per point, example:
        # { "x" : 0, "y" : 100, "color" : 'green', "marker" : 'o', "markerfacecolor" : 'blue', "markersize" : 12 }
        ]

################
### barchart ###
################
colormapping = None


######################
### clusterstacked ###
######################

# ID of the column that holds xtick labels for each element of the cluster
xticks_per_bar_id = 1

# number of clustered bars for clusterstacked charts
num_clustered = 4

# print sublabels per cluster
do_sublabels = True

# xtick labels for each element of the cluster
labels_rotation='horizontal'    # rotation
labels_y=-0.08                  # placement w.r.t y=0
xticks_y = -0.05


###############
### stacked ###
###############




#################
### linechart ###
#################
lineargs={
    "linewidth"     : 1,
    "markersize"    : 6
    }
line_styles = ('-', '--', '-.', ':', 'steps', ' ')
do_labels = True
do_x_as_xticks = False
line_label_enable = False

linecolors = [colorConverter.to_rgb(a) for a in (bw_palette_2).split()]


################
### roofline ###
################
num_points = 6
