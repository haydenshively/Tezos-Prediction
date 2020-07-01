import cv2
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt

from training.train_cnn_timeseries import MyData


def main(model_dir, data_dir):
    BATCH_SIZE = 16
    N_SAMPLES_IN = 40  # divide by 10 to get # hours the sequence covers
    N_SAMPLES_OUT = 5
    PROB_DISTRIB = 1

    generator = MyData(
        data_dir,
        BATCH_SIZE,
        N_SAMPLES_IN,
        N_SAMPLES_OUT,
        PROB_DISTRIB
    )

    price_true = generator.history.copy()
    price_true = price_true[:len(generator) * BATCH_SIZE + N_SAMPLES_IN, 8]
    price_true -= price_true.min()
    price_true /= price_true.max()

    cnn = models.load_model(model_dir)
    price_pred = cnn.predict(generator)
    price_pred_roll = np.zeros((sum(price_pred.shape), N_SAMPLES_OUT))

    price_pred_roll[:price_pred.shape[0]] = price_pred.copy()
    for i in range(1, N_SAMPLES_OUT):
        price_pred_roll[:, i] = np.roll(price_pred_roll[:, i], i)

    for i in range(N_SAMPLES_OUT):
        plt.plot(price_pred[:, i], label='Based on {} mins ago'.format(i*6+6))

    plt.legend()
    plt.title('Prediction Consistency over Time')
    plt.ylabel('k*Price')
    plt.xlabel('Time (6 minute increments)')
    plt.show()


if __name__ == '__main__':
    print(main('models/cnn_timeseries_16_40_5_1.h5', 'dataset/test'))
