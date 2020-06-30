from tensorflow.keras import layers, models

from models.model import Model


class LSTM(Model):
    def __init__(self, n_samples_in, n_samples_out, n_features_in, n_features_out):
        super().__init__()
        self.n_samples_in = n_samples_in
        self.n_samples_out = n_samples_out
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out

    def build(self):
        inputs = layers.Input(shape=(self.n_samples_in, self.n_features_in))

        conv1 = layers.Conv1D(16, 3)(inputs)
        conv2 = layers.Conv1D(32, 3)(conv1)
        conv3 = layers.Conv1D(64, 3)(conv2)

        lstm1 = layers.LSTM(128, return_sequences=True)(conv3)
        bn1 = layers.BatchNormalization()(lstm1)

        lstm2 = layers.LSTM(64)(bn1)
        bn2 = layers.BatchNormalization()(lstm2)

        dense1 = layers.Dense(32)(bn2)
        dense2 = layers.Dense(self.n_samples_out * self.n_features_out, activation='linear')(dense1)
        outputs = layers.Reshape((self.n_samples_out, self.n_features_out))(dense2)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        super().build()

    def compile(self):
        self.model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
        super().compile()
