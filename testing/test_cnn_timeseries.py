from tensorflow.keras import models

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

    cnn = models.load_model(model_dir)
    cnn.summary()
    return cnn.evaluate(generator, verbose=1)


if __name__ == '__main__':
    print(main('../models/cnn_timeseries_16_40_5_1.h5', '../dataset/test'))
