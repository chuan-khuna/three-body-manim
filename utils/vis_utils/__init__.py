# -30k theme

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patheffects
import seaborn as sns
import warnings
import os

from .utils import *

warnings.filterwarnings('ignore')


################################
# intialise default theme
################################

from .palettes import nzk

use_style('nzk')
default_palette = list(nzk.main_colors.values())
set_seaborn_palette(default_palette)


################################
# constants
################################

font_mono = "Incosolata"
font_serif = "Canela Text"
font_sans = "Gill Sans"
arrows = {'left': '←', 'right': '→', 'up': '↑', 'down': '↓'}
