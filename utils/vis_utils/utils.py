import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patheffects
import seaborn as sns
import os


# from typing import TypeVar
# PathEffect = TypeVar('PathEffect')


def get_path_effects(
    linewidth: float = 2.5, foreground: str = '#FAF7F4', alpha: float = 1.0, **kwargs
) -> list[patheffects.AbstractPathEffect]:
    """return [border_path_effect]"""
    return [
        patheffects.withStroke(linewidth=linewidth, foreground=foreground, alpha=alpha, **kwargs)
    ]


def use_style(style_name: str):
    """set matplotlib style from .mplstyle file in vis_utils/styles/"""
    plt.style.use(
        os.path.join(os.path.dirname(__file__), '..', f"vis_utils/styles/{style_name}.mplstyle")
    )


def set_style(style_name: str):
    """set matplotlib style from .mplstyle file in vis_utils/styles/"""
    use_style(style_name)


def mpl_import_fonts(font_paths: list[str]):
    for font in matplotlib.font_manager.findSystemFonts(font_paths):
        matplotlib.font_manager.fontManager.addfont(font)


def set_font(font_name: str):
    # set font
    matplotlib.rcParams['font.family'] = font_name


def set_seaborn_palette(colours: list[str]):
    sns.set_palette(colours)
