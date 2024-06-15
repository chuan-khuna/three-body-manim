import numpy as np
import wordcloud
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def generate_wordcloud(tokenised_texts, colormap: list = None) -> wordcloud.WordCloud:
    """Generate wordcloud object

    Args:
        tokenised_texts (_type_): list of [list of tokens]-like, pd.Series
        colormap (list, optional): list of string hex. Defaults to None.

    Returns:
        _type_: _description_
    """

    words, counts = np.unique(np.concatenate(list(tokenised_texts)), return_counts=True)
    freq_dict = {}

    for k, v in zip(words, counts):
        freq_dict[k] = v

    # check colour map variable
    if colormap is None:
        wordcloud_cmap = matplotlib.colors.ListedColormap(
            ["#202f66", "#4e3986", "#774199", "#a1479c", "#c84e8c", "#e85b6f", "#ff7048"])
    else:
        assert isinstance(colormap, list)
        wordcloud_cmap = matplotlib.colors.ListedColormap(colormap)

    wc = wordcloud.WordCloud(
        font_path="./fonts/Noto_Sans_Thai/NotoSansThai-VariableFont_wdth,wght.ttf",
        width=2000,
        height=1600,
        max_words=200,
        prefer_horizontal=1.0,
        background_color='white',
        colormap=wordcloud_cmap)
    wc.generate_from_frequencies(freq_dict)

    # plt.imshow(wc)
    # plt.axis(False)
    # plt.show()

    return wc
