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


def RREF(A):
    rows, cols = A.shape

    N = min(*A.shape)

    # np.eye(rows) will hold the cumulative transformation due to the elementary row operations
    augmented = np.concatenate((A.copy(), np.eye(rows)), 1)
    for j in range(N):
        pivot = augmented[j, j]

        if abs(pivot) < 1e-14:  # avoid dividng by really small non-zero value
            pivot = 0

        if pivot == 0.0:  # swap with another row below
            able_to_swap = False
            for k in range(j + 1, rows):
                if augmented[k, j] != 0.0:  # swap rows
                    tmp = augmented[j].copy()
                    augmented[j] = augmented[k]
                    augmented[k] = tmp
                    able_to_swap = True
                    break
            if not able_to_swap:
                # pivot & everything underneath are 0; skip to next pivot
                continue

        assert augmented[j, j] != 0.0

        if j == 2:  # breakpoint
            pass

        checkAugmented(augmented, A)  # checking for correctness
        augmented[j] /= augmented[j, j]  # make pivot location 1
        checkAugmented(augmented, A)  # checking for correctness
        pivot_row = augmented[j]
        checkAugmented(augmented, A)  # checking for correctness
        pivot = pivot_row[j]
        checkAugmented(augmented, A)  # checking for correctness

        for i in range(0, rows):  # going down the coulumn
            if i == 0 and j == 2:  # breakpoint
                pass

            if i != j:
                row_to_reduce = augmented[i, :]
                entry_to_zero = row_to_reduce[j]
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
