import os
import numpy as np
import matplotlib.pyplot as plt

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

truth = history

predi = np.load('./PredictedUpTo1.npy')

print(predi.shape)
plt.plot(predi[:, 8], '-b')
plt.plot(truth[:predi.shape[0], 8], '-r')
plt.show()
