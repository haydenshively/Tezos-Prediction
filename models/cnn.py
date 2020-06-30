from tensorflow.keras import layers, models, optimizers

from models.model import Model


class CNN(Model):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self):
        inputs = layers.Input(shape=self.input_shape)

        conv1 = layers.Conv2D(32, 3, activation='relu')(inputs)
        maxp1 = layers.MaxPool2D((2, 2), strides=(2, 2))(conv1)
        drop1 = layers.Dropout(0.25)(maxp1)

        conv2 = layers.Conv2D(64, 3, activation='relu')(drop1)
        conv3 = layers.Conv2D(64, 3, activation='relu')(conv2)
        maxp2 = layers.MaxPool2D((2, 2), strides=(2, 2))(conv3)
        drop2 = layers.Dropout(0.25)(maxp2)

        conv4 = layers.Conv2D(128, 3, activation='relu')(drop2)
        conv5 = layers.Conv2D(128, 3, activation='relu')(conv4)
        maxp3 = layers.MaxPool2D((2, 2), strides=(2, 2))(conv5)

        flat = layers.Flatten()(maxp3)
        dense = layers.Dense(self.output_shape, activation='linear')(flat)

        self.model = models.Model(inputs=inputs, outputs=dense)
        super().build()

    def compile(self):
        self.model.compile(optimizer=optimizers.SGD(0.001), loss='mse', metrics=['mae', 'accuracy'])
        super().compile()
