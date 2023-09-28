import numpy as np
"""----------------------PLA算法----------------------"""


def pla(x, y, max_it):
    [n_x, d_x] = x.shape
    it = 0
    w = np.zeros(d_x)

    while it < max_it:
        misclassified = False

        for k in range(n_x):
            if y[k] * np.dot(w, x[k]) <= 0:
                misclassified = True

                w += y[k] * x[k]
        if not misclassified:
            break

        it += 1

    return w
