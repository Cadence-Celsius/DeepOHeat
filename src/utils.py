import numpy as np
import matplotlib

cmap_data = {
    "red": np.array(
        [
            [0.0, 1.0, 1.0],
            [0.03, 1.0, 1.0],
            [0.215, 1.0, 1.0],
            [0.4, 0.0, 0.0],
            [0.586, 0.0, 0.0],
            [0.77, 0.0, 0.0],
            [0.954, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ),
    "green": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.215, 1.0, 1.0],
            [0.4, 1.0, 1.0],
            [0.586, 1.0, 1.0],
            [0.77, 0.8, 0.8],
            [0.954, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ),
    "blue": np.array(
        [
            [0.0, 0.16, 0.16],
            [0.03, 0.0, 0.0],
            [0.215, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.586, 1.0, 1.0],
            [0.77, 1.0, 1.0],
            [0.954, 1.0, 1.0],
            [1.0, 0.8, 0.8],
        ]
    ),
}


class MyCmap:

    @staticmethod
    def get_cmap():
        return matplotlib.colors.LinearSegmentedColormap(
            name="cmap", segmentdata=cmap_data
        ).reversed()
