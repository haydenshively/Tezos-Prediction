"""
TRAIN
"""
import os

from scipy import stats
import numpy as np
from tensorflow.keras.utils import Sequence

from transforms import GAF
from models import CNNTimeSeries


class MyData(Sequence):
    def __init__(self, dir, batch_size, n_samples_in, n_samples_out=1, distrib_size=1):

        history_chunks = []

        files = os.listdir(dir)
        files.sort()
        for file in files:
            filename = os.fsdecode(file)
            if filename.endswith('.npy'):
                history_chunks.append(np.load(os.path.join(dir, filename)))

        # Combine all chunks into one big history
        self.history = np.vstack(history_chunks)
        # Reshape so that different currencies aren't in separate channels
        shape = self.history.shape
        self.history = self.history.reshape((shape[0], shape[1] * shape[2]))
        # Save other info to instance
        self.extremes = [self.history[:, 8].min(), self.history[:, 8].max()]
        self.batch_size = batch_size
        self.n_samples_in = n_samples_in
        self.n_samples_out = n_samples_out
        self.distrib_size = distrib_size
        assert (self.n_samples_out == 1 or self.distrib_size == 1)

    def __len__(self):
        return (self.history.shape[0] - self.n_samples_in - self.n_samples_out) // self.batch_size - 2

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size

        X = np.zeros((self.batch_size, self.n_samples_in, self.n_samples_in, 1))
        Y = np.zeros((self.batch_size, self.n_samples_out if self.n_samples_out > 1 else self.distrib_size))

        for offset in range(self.batch_size):
            series_x_0 = batch_start + offset
            series_x_n = batch_start + offset + self.n_samples_in
            out_n = self.n_samples_out

            series = self.history[series_x_0:(series_x_n + out_n), 8]
            gaf_series = GAF(series[:-out_n], extremes=[series.min(), series.max()])
            gaf_out = GAF(series)

            X[offset] = np.expand_dims(gaf_series.encoded, -1)

            if self.n_samples_out > 1:
                Y[offset] = gaf_out.series[-out_n:]
            else:
                norm = stats.norm(loc=gaf_out.series[-1], scale=.4)
                Y[offset] = norm.pdf(np.linspace(-1.0, 1.0, self.distrib_size))

        return X, Y


if __name__ == '__main__':

    DATA_DIR = '../dataset/train'
    BATCH_SIZE = 16
    N_SAMPLES_IN = 40  # divide by 10 to get # hours the sequence covers
    N_SAMPLES_OUT = 5
    PROB_DISTRIB = 1

    generator = MyData(
        DATA_DIR,
        BATCH_SIZE,
        N_SAMPLES_IN,
        N_SAMPLES_OUT,
        PROB_DISTRIB
    )

    cnn = CNNTimeSeries(
        (N_SAMPLES_IN, N_SAMPLES_IN, 1),
        N_SAMPLES_OUT if N_SAMPLES_OUT > 1 else PROB_DISTRIB,
        BATCH_SIZE
    )
    cnn.build()
    cnn.compile()

    cnn.model.fit(generator,
                  epochs=6,
                  shuffle=True,
                  verbose=1,
                  steps_per_epoch=len(generator))

    cnn.model.save('cnn_%d_%d_%d_%d.h5' % (
        BATCH_SIZE, N_SAMPLES_IN, N_SAMPLES_OUT, PROB_DISTRIB
    ))
