"""
TRAIN
"""
import os
import math
import numpy as np
from tensorflow.keras.utils import Sequence


class OurData(Sequence):
    def __init__(self, dir, batch_size, n_samples_in):

        history_chunks = []

        files = os.listdir(dir)
        files.sort()
        for file in files:
            filename = os.fsdecode(file)
            if filename.endswith('.npy'):
                history_chunks.append(np.load(os.path.join(dir, filename)))

        # Combine all chunks into one big history
        self.history = np.vstack(history_chunks)
        # Normalize values to be between 0 and 1
        self.history = self.history - self.history.min(axis=0)
        maxes = self.history.max(axis=0)
        maxes[maxes == 0] = 1
        self.history = self.history / maxes
        # Reshape so that different currencies aren't in separate channels
        shape = self.history.shape
        self.history = self.history.reshape((shape[0], shape[1] * shape[2]))
        # Save other info to instance
        self.batch_size = batch_size
        self.n_samples_in = n_samples_in

    def __len__(self):
        return math.ceil(
            (self.history.shape[0] - self.n_samples_in - 15) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size

        X = np.zeros((self.batch_size, self.n_samples_in, self.history.shape[-1]))
        Y = np.zeros((self.batch_size, self.history.shape[-1]))

        for i in range(self.batch_size):
            X[i] = self.history[start + i: start + i + self.n_samples_in]
            Y[i] = self.history[start + i + self.n_samples_in]

        return X, Y


if __name__ == '__main__':
    from model_lstm import LSTM

    DATA_DIR = './train'
    BATCH_SIZE = 4
    N_SAMPLES_IN = 20

    generator = OurData(DATA_DIR, BATCH_SIZE, N_SAMPLES_IN)

    lstm = LSTM(N_SAMPLES_IN, 12, 12)
    lstm.build()
    lstm.compile()

    lstm.model.fit(generator,
                   epochs=3,
                   verbose=1,
                   steps_per_epoch=generator.__len__())

    lstm.model.save('trained_lstm.h5')
