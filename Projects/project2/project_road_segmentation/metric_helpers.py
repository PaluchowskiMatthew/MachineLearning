import numpy as np
def compute_true_positive_rate(groundtruth, prediction):
    # Get non-zeros in prediction and grountruth arrays
    Zn = np.nonzero(prediction)[0]
    Yn = np.nonzero(groundtruth)[0]

    TPR = len(list(set(Yn) & set(Zn))) / float(len(prediction))
    return TPR
