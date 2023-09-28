import numpy as np
import random
"""----------------------pocket算法----------------------"""


def errorindex(x, w, y):
    error_indexes = []

    for j in range(len(x)):
        if np.dot(x[j], w) * y[j] <= 0:
            error_indexes.append(j)
    return error_indexes


def pocket(x, y, it, max_it_no_change):
    w = np.zeros(x.shape[1])
    errors = errorindex(x, w, y)
    it_no_change = 0

    for j in range(it):
        if len(errors) == 0:
            return w
        else:
            pick_index = random.choice(errors)
            temp_w = w + y[pick_index] * x[pick_index]
            temp_errors = errorindex(x, temp_w, y)
            if len(temp_errors) <= len(errors):
                w = temp_w
                errors = temp_errors
            else:
                it_no_change += 1
        if it_no_change >= max_it_no_change:
            break
    return w
