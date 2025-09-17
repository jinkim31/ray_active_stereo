import pickle
import os

def pickle_get(path, functor):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        result = functor()
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result