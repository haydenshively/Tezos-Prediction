import numpy as np


class GAF:
    def __init__(self, series, extremes=[None, None]):
        # MARK: - Scaling
        # convert list to ndarray
        series = np.array(series).copy()
        assert (len(series.shape) == 1)
        self.length = series.shape[0]
        # compute min and max of series
        extremes[0] = series.min() if extremes[0] is None else min(series.min(), extremes[0])
        extremes[1] = series.max() if extremes[1] is None else max(series.max(), extremes[1])
        self.extremes = extremes
        # scale series between -1 and 1
        series *= 2.0
        series -= sum(self.extremes)
        series /= self.extremes[1] - self.extremes[0]
        # correct for floating point errors
        self.series = np.clip(series, -1.0, 1.0)

        # MARK: - Polar Encoding
        phi = np.arccos(self.series)
        x = np.meshgrid(phi, phi)
        self.encoded = np.cos(x[0] + x[1])
