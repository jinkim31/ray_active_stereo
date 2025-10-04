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


def sanitize_lists(lists):
    # Assume all sublists are same length
    if not lists:
        return lists

    n = len(lists[0])  # number of elements in each sublist

    # Keep only indices where all sublists have non-None
    valid_indices = [i for i in range(n) if all(sub[i] is not None for sub in lists)]

    # Build filtered lists
    return [[sub[i] for i in valid_indices] for sub in lists]