import numpy as np


def checkAugmented(augmented, A):
    cols = A.shape[1]
    tolerance = 1e-10
    max_tolerable_error = tolerance * abs(A).max()
    max_error = abs(augmented[:, cols:] @ A - augmented[:, :cols]).max()
    if max_tolerable_error < max_error:
        print(f"max_error: {max_error}")
        print(f"max_tolerable_error: {max_tolerable_error}")
        print(augmented, A)
        exit(1)


def RREF(A, col_idxs):
    rows, cols = A.shape

    ZERO_THRESHHOLD = 1e-14

    # np.eye(rows) will hold the cumulative transformation due to the elementary row operations
    augmented = np.concatenate((A.copy(), np.eye(rows)), 1)
    for pivot_row_idx, pivot_col_idx in enumerate(col_idxs):
        pivot = augmented[pivot_row_idx, pivot_col_idx]

        if abs(pivot) < ZERO_THRESHHOLD:  # avoid dividng by really small non-zero value
            pivot = 0

        if pivot == 0.0:  # swap with another row below
            able_to_swap = False
            for idx_of_row_to_swap_with in range(pivot_row_idx + 1, rows):
                candidate_pivot = augmented[idx_of_row_to_swap_with, pivot_col_idx]
                if ZERO_THRESHHOLD < abs(candidate_pivot):  # swap rows
                    tmp = augmented[pivot_row_idx].copy()
                    augmented[pivot_row_idx] = augmented[idx_of_row_to_swap_with]
                    augmented[idx_of_row_to_swap_with] = tmp
                    able_to_swap = True
                    break
            if not able_to_swap:
                # pivot & everything underneath are 0; skip to next pivot
                continue

        assert augmented[pivot_row_idx, pivot_col_idx] != 0.0

        checkAugmented(augmented, A)  # checking for correctness
        augmented[pivot_row_idx] /= augmented[pivot_row_idx, pivot_col_idx]
        checkAugmented(augmented, A)  # checking for correctness
        pivot_row = augmented[pivot_row_idx]
        pivot = pivot_row[pivot_col_idx]
        checkAugmented(augmented, A)  # checking for correctness

        for i in range(0, rows):  # going down the column
            if i != pivot_row_idx:
                row_to_reduce = augmented[i, :]
                entry_to_zero = row_to_reduce[pivot_col_idx]
                if entry_to_zero == 0.0:
                    continue
                else:
                    # TODO divide by 0?
                    row_to_reduce -= (entry_to_zero / pivot) * pivot_row

            checkAugmented(augmented, A)  # checking for correctness

    rref = augmented[:, :cols]
    E = augmented[:, cols:]  # cumulative transformation due to elementary row ops.
    return rref, E


if __name__ == "__main__":
    # np.set_printoptions(precision=99999, suppress=True, linewidth=999999)
    np.set_printoptions(precision=99999, linewidth=999999)

    A = np.array(
        [
            [-6, 3, 1],
            [4, -6, -10],
            [-1, -3, -8],
        ]
    )
    rref, E = RREF(A)

    tolerance = 1e-14
    if abs(E @ A - rref).max() > tolerance:
        print(A)
