import os

import numpy as np
from tensorflow.keras.utils import Sequence

from transforms import GAF
from models import LSTM


class MyData(Sequence):
    def __init__(self, dir, batch_size, n_samples_in, n_samples_out=1):

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

    def __len__(self):
        return (self.history.shape[0] - self.n_samples_in - self.n_samples_out) // self.batch_size - 2

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size

        X = np.zeros((self.batch_size, self.n_samples_in))
        Y = np.zeros((self.batch_size, self.n_samples_out))

        for offset in range(self.batch_size):
            series_x_0 = batch_start + offset
            series_x_n = batch_start + offset + self.n_samples_in
            out_n = self.n_samples_out

            series = self.history[series_x_0:(series_x_n + out_n), 8]
            # Just using GAF to scale to [-1, 1]
            gaf = GAF(series)

            X[offset] = gaf.series[:-out_n]
            Y[offset] = gaf.series[-out_n:]

        return X, Y


def main(data_dir):
    BATCH_SIZE = 16
    N_SAMPLES_IN = 40  # divide by 10 to get # hours the sequence covers
    N_SAMPLES_OUT = 5

    generator = MyData(
        data_dir,
        BATCH_SIZE,
        N_SAMPLES_IN,
        N_SAMPLES_OUT
    )

    lstm = LSTM(N_SAMPLES_IN, N_SAMPLES_OUT, 1, 1)
    lstm.build()
    lstm.compile()

    lstm.model.fit(generator,
                   epochs=6,
                   shuffle=True,
                   verbose=1,
                   steps_per_epoch=len(generator))

    lstm.model.save('models/lstm_%d_%d_%d.h5' % (
        BATCH_SIZE, N_SAMPLES_IN, N_SAMPLES_OUT
    ))


if __name__ == '__main__':
    main('../dataset/train')
