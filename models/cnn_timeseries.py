import numpy as np

from transforms.gaf import GAF
from models.cnn import CNN


class CNNTimeSeries(CNN):
    def __init__(self, image_shape, prediction_shape, batch_size, extremes=[None, None]):
        super().__init__(image_shape, prediction_shape)
        self.batch_size = batch_size
        self.extremes = extremes
        self.dataX = []
        self.dataY = []
        self.isReadyToTrain = False

    def _preprocess(self, sequence):
        gaf = GAF(sequence, self.extremes)
        self.extremes = gaf.extremes
        return [gaf.encoded, gaf.series[0]]

    def record(self, sequence):
        preprocessed = self._preprocess(sequence)
        if len(self.dataX) > 0:
            # if any input (GAF image) exists in `dataX`, we assume that its
            # corresponding output (float) is the first value in `sequence`
            self.dataY.append(preprocessed[1])

        # append a new GAF image to `dataX`
        self.dataX.append(preprocessed[0])
        length = len(self.dataX)
        if length > self.batch_size:
            # if we've stored more than 1 batch's worth of images,
            # remove the oldest one
            self.dataX.pop(0)
            # if `isReadyToTrain` is already true, self must be the second
            # time we've run through self conditional, meaning `dataY`
            # is ready to be shifted as well
            if self.isReadyToTrain:
                self.dataY.pop(0)
            else:
                self.isReadyToTrain = True

    def train(self):
        if self.isReadyToTrain:
            x = np.concatenate(self.dataX)
            x = np.expand_dims(x, -1)
            y = np.array(self.dataY)

            return self.model.train_on_batch(x, y)

    def predict_next_value_in(self, sequence):
        sequence = np.expand_dims(self._preprocess(sequence), [0, -1])
        prediction = self.model.predict(sequence)
        return prediction * self.extremes[1] + self.extremes[0]

    def predict_from_record(self):
        sequence = np.expand_dims(self._preprocess(self.dataX[-1]), [0, -1])
        prediction = self.model.predict(sequence)
        return prediction * self.extremes[1] + self.extremes[0]
