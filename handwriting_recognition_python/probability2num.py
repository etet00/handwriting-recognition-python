import numpy as np


def probability2num(predictions):
    pre_out = []
    for prediction in predictions:
        pre_out.append(np.argmax(prediction))
    return pre_out
