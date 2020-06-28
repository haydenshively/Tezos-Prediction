import os
import numpy as np

def num_files_in(dir):
    return len(next(os.walk(dir))[2])

def build_dataset_from(dir, cache = None):
    if cache is not None:
        try:
            return np.load(cache)
        except:
            pass

    dir_enc = os.fsencode(dir)
    X0 = np.load(os.path.join(dir, os.fsdecode(os.listdir(dir_enc)[0])))
    X = np.zeros((num_files, X0.shape[0], X0.shape[1]))

    for i, file in enumerate(os.listdir(directory)):
         filename = os.fsdecode(file)
         if filename.endswith(".npy"):
             X[i] = np.load(os.path.join(dir, filename))

    if cache is not None:
        np.save(cache, X)

    return X
