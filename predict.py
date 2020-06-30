if __name__ == '__main__':
    import os
    import numpy as np
    from tensorflow.keras import models

    dir = 'dataset/test'

    history_chunks = []

    files = os.listdir(dir)
    files.sort()
    for file in files:
         filename = os.fsdecode(file)
         if filename.endswith('.npy'):
             history_chunks.append(np.load(os.path.join(dir, filename)))

    # Combine all chunks into one big history
    history = np.vstack(history_chunks)
    # Normalize values to be between 0 and 1
    history = history - history.min(axis=0)
    maxes = history.max(axis = 0)
    maxes[maxes == 0] = 1
    history = history / maxes
    # Reshape so that different currencies aren't in separate channels
    shape = history.shape
    history = history.reshape((shape[0], shape[1]*shape[2]))


    lstm = models.load_model('trained_lstm.h5')

    x = history[:20]
    y = history.copy()
    for i in range(20, y.shape[0]):

        pred = lstm.predict(x[None, :, :])[0]
        print(pred.shape)
        y[i] = pred

        x = np.roll(x, -1)
        x[-1] = pred

        perc_error = (x - history[i-19 : i+1]).mean()
        if abs(perc_error) > 1.0:
            break
        print('{}% Error after {} hours'.format(int(perc_error*100), i/10.0/60.0))

    np.save('PredictedUpTo1.npy', y)
