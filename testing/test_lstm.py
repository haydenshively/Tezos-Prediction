from tensorflow.keras import models

from training.train_lstm import MyData


def main(model_dir, data_dir):
    BATCH_SIZE = 16
    N_SAMPLES_IN = 40  # divide by 10 to get # hours the sequence covers
    N_SAMPLES_OUT = 5

    generator = MyData(
        data_dir,
        BATCH_SIZE,
        N_SAMPLES_IN,
        N_SAMPLES_OUT
    )

    lstm = models.load_model(model_dir)
    lstm.summary()
    return lstm.evaluate(generator, verbose=1)


if __name__ == '__main__':
    print(main('../models/lstm_16_40_5.h5', '../dataset/test'))
