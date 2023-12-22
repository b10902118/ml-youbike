import numpy as np
from utils import error


# def error(y_true: np.ndarray, y_pred: np.ndarray, tots: np.ndarray) -> np.float64:
#    return (
#        3
#        * np.abs((y_pred - y_true) / tots)
#        * (np.abs(y_true / tots - 1 / 3) + np.abs(y_true / tots - 2 / 3))
#    ).mean()


def first_greater_prefix_sum_idx(arr, target):
    prefix_sum = 0

    for i, element in enumerate(arr):
        prefix_sum += element

        if prefix_sum > target:
            return i


# error when nan
def optimal_median(y_true: np.ndarray, tot: int) -> tuple[np.float64, np.float64]:
    # assert len(y_true.shape) == 1, "optimal_median: shape error"
    # print(f"y_true: {y_true}")

    nan_indices = np.isnan(y_true)

    # Use boolean indexing to drop NaN values
    y_true = y_true[~nan_indices]

    arr_len = y_true.shape[0]
    tots = np.full(arr_len, tot)

    # generalized median
    y_sorted = np.sort(y_true)
    weight = np.abs(y_sorted / tot - 1 / 3) + np.abs(y_sorted / tot - 2 / 3)
    w_mid = np.sum(weight) / 2
    w_cur = 0

    # if odd, first > w_mid
    # if even, before m1 must less than w_mid, m2 must greater than w_mid
    best_sbi = y_sorted[first_greater_prefix_sum_idx(weight, w_mid)]
    best_err = error(y_true, np.full(arr_len, best_sbi), tots)

    return best_sbi, best_err

    # tested with the following
    # for sbi in np.arange(0, tot, step):
    #    sbis = np.full(arr_len, sbi)
    #    err = error(y_true, sbis, tots)
    #    if err < best_err:
    #        # if best_err - err > 0.00000001:
    #        # print("predict not optimal")
    #        # print(f"{best_sbi}: {best_err}")
    #        # print(f"{sbi}: {err}")
    #        best_sbi, best_err = sbi, err

    # return (best_sbi, best_err / arr_len)
