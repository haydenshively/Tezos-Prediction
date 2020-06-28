from tensorflow.keras import layers, models

from model import Model


class LSTM(Model):
    def __init__(self, n_samples_in, n_features_in, n_classes_out):
        super().__init__()
        self.n_samples_in = n_samples_in
        self.n_features_in = n_features_in
        self.n_classes_out = n_classes_out

    def build(self):
        inputs = layers.Input(shape=(self.n_samples_in, self.n_features_in))

        conv1 = layers.Conv1D(16, 3)(inputs)
        conv2 = layers.Conv1D(32, 3)(conv1)
        conv3 = layers.Conv1D(64, 3)(conv2)
        lstm1 = layers.LSTM(128, return_sequences=True)(conv3)
        # bn1 = layers.BatchNormalization()(lstm1)
        lstm2 = layers.LSTM(64)(lstm1)
        # bn2 = layers.BatchNormalization()(lstm2)
        dense1 = layers.Dense(32)(lstm2)
        outputs = layers.Dense(self.n_classes_out)(dense1)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        super().build()

    def compile(self):
        self.model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
        super().compile()
